import argparse
import math
import os

import torch
from tqdm import tqdm
from transformers import set_seed

import wandb
from models import build_model
from tasks import get_task_and_datasets

from .utils import set_optimizer_scheduler


def main():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--task", type=str, default="arithmetic")
    parser.add_argument("--input_size", type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT", "TMLT"])
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_loop", type=int, default=16)
    parser.add_argument("--is_causal", action="store_true")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()
    print(args)

    seed = 42
    set_seed(seed)
    os.makedirs("./output", exist_ok=True)

    task, train_dataset, test_dataset = get_task_and_datasets(args)

    print(task.config)

    collate_fn = getattr(task, "collate_fn", None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    from .curriculum import GeometricIncreaseCurriculum, RegularIncreaseCurriculum

    max_length = args.input_size
    # max_length = task.config["max_length"]
    initial_len = task.config.get("min_input_size", 4)
    if args.task == "word" or args.task == "cvp":
        total_steps = len(train_loader) * args.epoch
        # initial_len = 4
        increase_amount = 2
        n_increments = math.ceil((max_length - initial_len) / increase_amount) + 1
        increase_frequency = max(1, total_steps // n_increments)
        curriculum = RegularIncreaseCurriculum(
            initial_sequence_length=initial_len,
            increase_frequency=increase_frequency,
            increase_amount=increase_amount,
            sample_all_length=False,
            max_sequence_length=max_length,
        )
    else:
        increase_factor = 2
        curriculum = GeometricIncreaseCurriculum(
            initial_sequence_length=initial_len,
            base_steps=40 * len(train_loader),  # S₀
            increase_factor=increase_factor,
            sample_all_length=False,
            max_sequence_length=max_length,
            warmup_steps=args.warmup * len(train_loader),
        )
    train_dataset.set_curriculum(curriculum)
    # test_dataset.set_curriculum(curriculum)

    # Model
    model = build_model(args, task)
    model.cuda()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=True)
        print(f"Loaded model from {args.model_path}")

    optimizer, scheduler = set_optimizer_scheduler(model, args, train_loader)

    wandb.init(project="CoT-vs-Loop", config=args, name=f"{args.task}_{args.input_size}_{args.model}")

    for epoch in range(args.epoch):
        model.train()
        seq_len = curriculum.sample_sequence_length()
        print(f"Epoch {epoch + 1}/{args.epoch}, Sequence Length: {seq_len}")
        for i, (input_ids, y) in enumerate(tqdm(train_loader)):
            inputs, y = input_ids.cuda(), y.long().cuda()
            logits = model(inputs)
            loss = task.pointwise_loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            curriculum.step()  # assume single GPU training
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                seq_len = curriculum.sample_sequence_length()
                wandb.log({"loss": loss.item(), "lr": lr})
                wandb.log({"input_size": seq_len})

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if args.task == "word" or args.task == "cvp":
                    total_acc = torch.zeros(task.config["max_length"], device="cpu")
                else:
                    total_acc = torch.tensor(0.0, device="cpu")
                for i, (input_ids, y) in enumerate(tqdm(test_loader)):
                    inputs, y = input_ids.cuda(), y.long().cuda()
                    logits = model(inputs)
                    acc = task.accuracy_fn(logits, y).detach().cpu()
                    total_acc += acc
            avg_acc = total_acc / len(test_loader)
            wandb.log({"test_accuracy": avg_acc})
            print(f"Epoch {epoch + 1}, Test Accuracy: {avg_acc}")

            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")


if __name__ == "__main__":
    main()
