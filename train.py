import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_scheduler, set_seed

import wandb
from model import LoopedGPT


def evaluate(cur_loader, max_iter=None):
    Sum, correct = 0, 0
    iter_count = 0

    for input_ids, y, _ in cur_loader:
        if max_iter and iter_count > max_iter:
            break
        inputs, y = input_ids.cuda(), y.cuda()
        logits = model(inputs)
        Sum += torch.as_tensor(inputs.shape[0]).cuda()

        truth = torch.where(y > 0, 1, 0)
        predict = torch.where(torch.argmax(logits, dim=2) == y, 1, 0) * truth
        correct += torch.sum(torch.where(torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0))

        iter_count += 1

    return correct / Sum


parser = argparse.ArgumentParser(description="train")

parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--file", type=str, default="Data")
parser.add_argument("--folder", type=str, default="arithmetic")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--drop", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="./output/log")
parser.add_argument("--wandb_name", type=str, default="CoT-vs-Loop")
parser.add_argument("--maxlen", type=int, default=120)
parser.add_argument("--maxdata", type=int, default=120)
parser.add_argument("--maxans", type=int, default=30)
parser.add_argument("--vocab", type=int, default=21)
parser.add_argument("--write2file", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--num_loop", type=int, default=10)
parser.add_argument("--head", type=int, default=4)
parser.add_argument("--num_range", type=int, default=11)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--chain", action="store_true", default=False)
parser.add_argument("--rpe", action="store_true", default=False)

args = parser.parse_args()
import sys

sys.path.append(args.folder)
from dataloader import getLoader  # type: ignore

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
main_process = 0
set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
# if dist.get_rank() == main_process:
#    log_writer = SummaryWriter(log_dir=args.output_dir)


def set_optimizer_scheduler(model):
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


# set up wandb
wandb.init(project="CoT-vs-Loop", config=args, name=args.wandb_name)

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
# dist.barrier()
model = LoopedGPT(args).cuda()
if args.model_path:
    model.load_state_dict(torch.load(args.model_path), strict=True)
# model = DDP(model, device_ids=[local_rank])
# model_without_ddp = model.module
optimizer, scheduler = set_optimizer_scheduler(model)
loader, testloader = getLoader(args)

criterion = nn.CrossEntropyLoss(ignore_index=0)
for epoch in range(args.epoch):
    model.train()
    # loader.sampler.set_epoch(epoch)
    pbar = tqdm(loader) if not args.write2file else loader
    for data_iter_step, (input_ids, y, _) in enumerate(pbar):
        inputs, y = input_ids.cuda(), y.long().cuda()
        logits = model(inputs)
        loss = criterion(logits.transpose(1, 2), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_1000x = int((data_iter_step / len(loader) + epoch) * 1000)
        if data_iter_step % 100 == 0:
            # wandb.log({"loss": loss.item()})
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({"loss": loss.item(), "lr": lr})
            if not args.write2file:
                pbar.set_description(f"epoch:{epoch}")

    scheduler.step()

    if (epoch + 1) % (args.epoch // 10) == 0:
        out_dir = os.path.join(args.output_dir, wandb.run.id)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{out_dir}/epoch_{epoch+1}.pt")

    if (epoch + 1) % (args.epoch // 10) == 0:
        model.eval()
        with torch.no_grad():
            train_acc = evaluate(loader, max_iter=100)
            print(f"train acc:{train_acc}")
            wandb.log({"train_acc": train_acc})

            acc = evaluate(testloader)
            print(f"test acc:{acc}")
            wandb.log({"test_acc": acc})
