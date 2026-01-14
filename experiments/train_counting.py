import argparse
import os
import random
from itertools import islice

import torch
from torch import optim
from tqdm import tqdm
from transformers import get_scheduler, set_seed

import wandb
from models import build_model
from tasks.counting.dnf import (
    CLAUSE_WIDTH,
    NUM_CLAUSES,
    NUM_VARS,
    DNFCountOfflineDataset,
    DNFCountOnlineDataset,
    DNFCountTask,
    gen_random_dnf,
)


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
        num_warmup_steps=1000,
        num_training_steps=steps_per_epoch * args.epoch,
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--task", type=str, default="dnf", choices=["dnf", "coloring"])
    parser.add_argument("--num_vars", type=int, default=NUM_VARS)
    parser.add_argument("--num_clauses", type=int, default=NUM_CLAUSES)
    parser.add_argument("--clause_width", type=int, default=CLAUSE_WIDTH)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT", "CT"])
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_loop", type=int, default=16)
    parser.add_argument("--n_step", type=int, default=4)
    parser.add_argument("--n_eval_loop", type=int, default=None)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--use_rope", action="store_true")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--chain", action="store_true")

    parser.add_argument("--num_mc_samples", type=int, default=100)

    args = parser.parse_args()
    print(args)

    assert args.task in ("dnf", "coloring")

    set_seed(42)
    os.makedirs("./output", exist_ok=True)

    if args.task == "dnf":
        n = args.num_vars
        m = args.num_clauses
        w = 3
        task = DNFCountTask(n, m, w, chain=args.chain)
        print(task.config)

        dummy_dnf = gen_random_dnf(n, m, w, random.Random())
        p_clause = [2.0 ** (-len(conj)) for conj in dummy_dnf]
        s = sum(p_clause)

    else:  # coloring
        # use num_vars as number of nodes, num_clauses as degree
        from tasks.counting.coloring import ColoringsDataset, ColoringTask

        n = args.num_vars
        d = args.num_clauses
        task = ColoringTask(n=n, d=d)
        print(task.config)

    if args.task == "dnf":
        if args.chain:
            train_dataset = DNFCountOnlineDataset(task.config, split="train", seed=args.seed)
            test_dataset = DNFCountOnlineDataset(task.config, split="test", seed=args.seed)
        else:
            train_dataset = DNFCountOfflineDataset(task.config, split="train", seed=args.seed)
            test_dataset = DNFCountOfflineDataset(task.config, split="test", seed=args.seed)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    else:
        # ColoringsDataset yields ((input_coloring, edge_index), target_coloring)
        from tasks.counting.coloring import ColoringsDataset

        train_dataset = ColoringsDataset(task.config, split="train")
        test_dataset = ColoringsDataset(task.config, split="test")

        def coloring_collate(batch):
            # batch: [ ((input_coloring, edge_index), target), ... ]
            inputs = [item[0][0] for item in batch]
            targets = [item[1] for item in batch]
            # stack
            return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, collate_fn=coloring_collate
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=coloring_collate)

    # Model
    model = build_model(args, task)
    model.cuda()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=True)
        print(f"Loaded model from {args.model_path}")

    steps_per_epoch = args.steps_per_epoch if args.chain else len(train_loader)

    optimizer, scheduler = set_optimizer_scheduler(model, args, steps_per_epoch=steps_per_epoch)

    wandb.init(project="cotloop", config=args, name=f"{args.task}_{args.num_vars}_{args.model}")

    total_epochs = 1 if args.chain else args.epoch
    for epoch in range(total_epochs):
        model.train()
        loader = islice(train_loader, steps_per_epoch) if args.chain else train_loader
        for i, (input_ids, y) in enumerate(tqdm(loader, total=steps_per_epoch if args.chain else len(train_loader))):
            inputs, y = input_ids.cuda(), y.long().cuda()

            # if args.model == "CT":
            #    inputs = inputs[:, -(epoch + 1) :]

            logits = model(inputs)
            loss = task.pointwise_loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": loss.item(), "lr": lr})

        n_test = 100
        tau = args.num_mc_samples
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epoch - 1:

            # save model
            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")

        # evaluate
        model.eval()
        with torch.no_grad():
            if args.task == "dnf":
                total_relative_error = 0.0
                loader = islice(test_loader, n_test) if args.chain else test_loader
                for input_ids, gt_count in tqdm(loader, total=n_test if args.chain else len(test_loader)):
                    inputs = input_ids.cuda()
                    N = None
                    if args.chain:
                        while N is None or N == 0:
                            N = 0
                            test_batch_size = 1
                            inputs = inputs.repeat(test_batch_size, 1)
                            for _ in range(tau // test_batch_size):
                                idx = model.generate(inputs, max_new_tokens=task.config["max_length"] - inputs.shape[1])
                                preds = idx[:, -2]
                                N += (preds == 5).sum().item()

                        mu_hat = (tau * s) / ((m * N) + 1e-8)
                        count_pred = mu_hat * (2.0**n)
                    else:
                        logit = model(inputs)
                        count_pred = torch.argmax(logit[0, -1]).item()
                        count_pred -= task.config["vocab_size"] - 2**n - 1

                    relative_error = abs(count_pred - gt_count.item()) / (gt_count.item() + 1e-8)
                    total_relative_error += relative_error
                avg_acc = total_relative_error / n_test
                print(f"Epoch {epoch + 1}, Test Accuracy: {avg_acc}")

            else:
                # TODO: sampling and histogram with total variation distance
                total_acc = 0.0
                n_test_samples = n_test
                loader = islice(test_loader, n_test_samples) if args.chain else islice(test_loader, n_test_samples)
                for input_ids, target in tqdm(loader, total=n_test_samples):
                    inputs = input_ids.cuda()
                    target = target.cuda()
                    logit = model(inputs)
                    preds = logit.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    total_acc += acc
                avg_acc = total_acc / n_test_samples
                print(f"Epoch {epoch + 1}, coloring per-node accuracy: {avg_acc}")


if __name__ == "__main__":
    main()
