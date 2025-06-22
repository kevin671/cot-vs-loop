import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_scheduler, set_seed

import wandb
from models.transformer import GPT, LoopedTF


def set_optimizer_scheduler(model, args) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
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
        name="linear", optimizer=optimizer, num_warmup_steps=args.warmup, num_training_steps=args.epoch
    )
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--task", type=str, default="arithmetic")
    parser.add_argument("--input_length", type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT"])
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_loop", type=int, default=10)

    # parser.add_argument("--curriculum", action="store_true", default=False)

    args = parser.parse_args()

    seed = 42
    set_seed(seed)
    os.makedirs("./output", exist_ok=True)

    # Task and Dataset
    task = hoge()

    datal_oader, test_loader = task.get_data_loaders()

    # Model
    if args.model == "Looped":
        model = LoopedTF(args).cuda()
    else:
        model = GPT(args).cuda()

    # if args.model_path:
    #    model.load_state_dict(torch.load(args.model_path), strict=True)

    optimizer, scheduler = set_optimizer_scheduler(model, args)

    # set up wandb
    wandb.init(project="CoT-vs-Loop", config=args, name=ae)

    for epoch in range(args.epoch):
        model.train()
        # loader.sampler.set_epoch(epoch)
        for i, (input_ids, y, _) in enumerate(tqdm(data_loader)):

            # ここもタスク依存にしたい
            # lossをどうやって取るべきか...
            inputs, y = input_ids.cuda(), y.long().cuda()
            # これはなんだ...
            if args.model_arch == "Looped":
                input_mask = (y != 0).cuda()
                inputs = inputs.masked_fill(~input_mask, 0)
            logits = model(inputs)
            loss = task.criterion(logits.transpose(1, 2), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": loss.item(), "lr": lr})

        scheduler.step()

        if (epoch + 1) % (args.epoch // 10) == 0:
            results = task.evaluate(model, test_loader)
            wandb.log(results)

            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            # torch.save(model.state_dict(), f"{out_dir}/epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")


if __name__ == "__main__":
    main()
