import argparse
import os

import torch
import torch.nn as nn
from transformers import set_seed

from model import LoopedGPT

parser = argparse.ArgumentParser(description="test")

parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--file", type=str, default="Data")
parser.add_argument("--folder", type=str, default="arithmetic")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--chain", action="store_true", default=False)
parser.add_argument("--rpe", action="store_true", default=False)
parser.add_argument("--maxlen", type=int, default=120)
parser.add_argument("--maxdata", type=int, default=120)
parser.add_argument("--maxans", type=int, default=30)
parser.add_argument("--vocab", type=int, default=21)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--drop", type=float, default=0.1)
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--num_loop", type=int, default=10)
parser.add_argument("--head", type=int, default=4)
parser.add_argument("--num_range", type=int, default=11)
parser.add_argument("--seed", type=int, default=2023)

args = parser.parse_args()

import sys

sys.path.append(args.folder)
from dataloader import getLoader  # type: ignore

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
main_process = 0
set_seed(2023)

model = LoopedGPT(args).cuda()
if args.model_path:
    model.load_state_dict(torch.load(args.model_path), strict=True)
_, test_loader = getLoader(args)

criterion = nn.CrossEntropyLoss(ignore_index=0)
if not args.chain:

    def evaluate(cur_loader):
        Sum, correct = 0, 0
        # cur_loader.sampler.set_epoch(0)
        for input_ids, y, _ in cur_loader:
            inputs, y = input_ids.cuda(), y.cuda()
            logits = model(inputs)
            Sum += torch.as_tensor(inputs.shape[0]).cuda()
            truth = torch.where(y > 0, 1, 0)
            predict = torch.where(torch.argmax(logits, dim=2) == y, 1, 0) * truth
            correct += torch.sum(torch.where(torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0))
        return correct / Sum

    model.eval()
    with torch.no_grad():
        acc = evaluate(test_loader)
        print(f"test acc:{acc}")

else:
    from eval import evaluate  # type: ignore

    model.eval()
    with torch.no_grad():
        acc = evaluate(model, test_loader)
        print(f"test acc:{acc}")
