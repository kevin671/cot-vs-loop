import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_scheduler, set_seed

import wandb
from models.transformer import GPT, LoopedTF, LoopedTFConfig


def set_optimizer_scheduler(
    model, args, dataloader
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=len(dataloader) * args.warmup,
        num_training_steps=len(dataloader) * args.epoch,
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--task", type=str, default="arithmetic")
    parser.add_argument("--input_length", type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT"])
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_loop", type=int, default=16)
    parser.add_argument("--is_causal", type=bool, default=True)

    # parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")

    # parser.add_argument("--curriculum", action="store_true", default=False)

    args = parser.parse_args()

    seed = 42
    set_seed(seed)
    os.makedirs("./output", exist_ok=True)

    # Task and Dataset
    from tasks.nc1.word import WordProblemDataset, WordProblemTask

    # TODO: args.input_lengthを引数にするように
    task = WordProblemTask()
    train_dataset = WordProblemDataset(task.config, split="train")
    test_dataset = WordProblemDataset(task.config, split="test")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    from .curriculum import RegularIncreaseCurriculum

    # from .curriculum import GeometricIncreaseCurriculum

    max_length = task.config["max_length"]
    total_steps = len(train_loader) * args.epoch
    initial_len = 4
    increase_amount = 2
    n_increments = math.ceil((max_length - initial_len) / increase_amount) + 1
    # initial_len = 2
    # increase_factor = 2
    # n_increments = math.ceil(math.log2(max_length / initial_len))

    increase_frequency = max(1, total_steps // n_increments)
    curriculum = RegularIncreaseCurriculum(  # GeometricIncreaseCurriculum(
        initial_sequence_length=initial_len,
        increase_frequency=increase_frequency,
        increase_amount=increase_amount,
        # increase_factor=increase_factor,
        sample_all_length=False,
        max_sequence_length=max_length,
        warmup_steps=args.warmup * len(train_loader),
    )
    train_dataset.set_curriculum(curriculum)
    # test_dataset.set_curriculum(curriculum)

    # Model
    if args.model == "Looped":
        model_args = LoopedTFConfig(
            block_size=task.config["max_length"],
            vocab_size=task.config["vocab_size"],
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=0.0,
            n_loop=args.n_loop,
            is_causal=args.is_causal,
        )
        model = LoopedTF(model_args).cuda()
    else:
        model = GPT(args).cuda()

    # if args.model_path:
    #    model.load_state_dict(torch.load(args.model_path), strict=True)

    optimizer, scheduler = set_optimizer_scheduler(model, args, train_loader)

    wandb.init(project="CoT-vs-Loop", config=args, name=f"{args.task}_{args.input_length}_{args.model}")

    for epoch in range(args.epoch):
        model.train()
        seq_len = curriculum.sample_sequence_length()
        print(f"Epoch {epoch + 1}/{args.epoch}, Sequence Length: {seq_len}")
        # loader.sampler.set_epoch(epoch)
        for i, (input_ids, y) in enumerate(tqdm(train_loader)):
            inputs, y = input_ids.cuda(), y.long().cuda()
            print(inputs, y, flush=True)
            # seq_len = curriculum.sample_sequence_length()
            logits = model(inputs)
            loss = task.pointwise_loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            curriculum.step()  # assume single GPU training
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": loss.item(), "lr": lr})

        if (epoch + 1) % 10 == 0:
            model.eval()
            # seq_len = curriculum.sample_sequence_length()
            with torch.no_grad():
                if args.task == "word":
                    total_acc = torch.zeros(task.config["max_length"], device="cpu")
                else:
                    total_acc = torch.tensor(0.0, device="cpu")
                # seq_len = curriculum.sample_sequence_length()
                # print(f"Evaluating at Sequence Length: {seq_len}")
                # total_acc = torch.tensor(0.0, device="cpu")
                for i, (input_ids, y) in enumerate(tqdm(test_loader)):
                    inputs, y = input_ids.cuda(), y.long().cuda()
                    logits = model(inputs)
                    acc = task.accuracy_fn(logits, y).detach().cpu()
                    total_acc += acc
                    # acc = task.accuracy_fn(logits, y)  # .detach().cpu()
                    # total_acc += acc.item()
            avg_acc = total_acc / len(test_loader)
            wandb.log({"test_accuracy": avg_acc})
            print(f"Epoch {epoch + 1}, Test Accuracy: {avg_acc}")

            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")


if __name__ == "__main__":
    main()
