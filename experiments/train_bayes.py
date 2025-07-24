import argparse
import os
from itertools import islice

import torch
from torch import optim
from tqdm import tqdm
from transformers import get_scheduler, set_seed

import wandb
from models import build_model


def set_optimizer_scheduler(model, args, steps_per_epoch=10_000):
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
        num_warmup_steps=10000,
        num_training_steps=steps_per_epoch * args.epoch,
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--task", type=str, default="bayes_net")
    parser.add_argument("--input_size", type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT", "TMLT"])
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_loop", type=int, default=16)
    parser.add_argument("--is_causal", action="store_true")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic sampling")
    parser.add_argument("--chain", action="store_true")

    args = parser.parse_args()
    print(args)

    assert args.task == "bayes_net", "This script is specifically for the Bayes Net task."
    assert args.is_causal is True, "Causal mask is required for Bayes Net task."

    seed = 42
    set_seed(seed)
    os.makedirs("./output", exist_ok=True)

    from tasks.sharp_p.bayes_net import BayesNetOnlineDataset, BayesNetTask

    task = BayesNetTask()
    train_dataset = BayesNetOnlineDataset(
        task.config, split="train", deterministic=args.deterministic, chain=args.chain
    )
    test_dataset = BayesNetOnlineDataset(task.config, split="test", deterministic=args.deterministic, chain=args.chain)

    print(task.config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = build_model(args, task)
    model.cuda()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=True)
        print(f"Loaded model from {args.model_path}")

    steps_per_epoch = 10000

    optimizer, scheduler = set_optimizer_scheduler(model, args, steps_per_epoch=steps_per_epoch)

    wandb.init(project="cotloop", config=args, name=f"{args.task}_{args.input_size}_{args.model}")

    for epoch in range(args.epoch):
        model.train()
        loader = islice(train_loader, steps_per_epoch)
        for i, (input_ids, y) in enumerate(tqdm(loader, total=steps_per_epoch)):
            inputs, y = input_ids.cuda(), y.long().cuda()
            if args.model == "Looped":
                logits = model(inputs, n_loop=1)[0]
            else:
                logits = model(inputs)
            loss = task.pointwise_loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": loss.item(), "lr": lr})

        n_eval = 100
        n_loop_eval = args.n_loop  # task.config["num_nodes"]
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                # total_acc = torch.tensor(0.0, device="cpu")
                # total_acc = torch.zeros(n_loop_eval, device="cpu")
                total_acc = torch.zeros(n_loop_eval, task.config["max_length"], device="cpu")
                for input_ids, y in islice(test_loader, n_eval):
                    inputs, y = input_ids.cuda(), y.long().cuda()
                    logits_list = model(inputs, n_loop=n_loop_eval)
                    for l, logits in enumerate(logits_list):
                        acc_vec = task.accuracy_fn(logits, y).detach().cpu()
                        total_acc[l] += acc_vec
                    # total_acc += acc
            avg_acc = total_acc / n_eval
            print(f"Epoch {epoch + 1}, Test Accuracy: {avg_acc}")
            # wandb.log({"test_accuracy": avg_acc})

            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")


if __name__ == "__main__":
    main()
