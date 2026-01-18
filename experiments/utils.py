import torch
from torch import optim
from transformers import get_scheduler


def set_optimizer_scheduler(model, args, dataloader) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
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
        # num_warmup_steps=len(dataloader) * args.warmup,
        num_warmup_steps=1000 if args.warmup > 0 else 0,
        num_training_steps=len(dataloader) * args.epoch,
    )
    return optimizer, scheduler


# https://github.com/facebookresearch/coconut/blob/main/utils.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random

import numpy as np
import torch


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint_adapt_wpe(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a checkpoint into `model`, adapting or skipping `transformer.wpe.weight`
    if its shape mismatches the current model.

    This function mutates the checkpoint state dict as needed and calls
    `model.load_state_dict(..., strict=False)`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    wpe_key = "transformer.wpe.weight"
    model_state = model.state_dict()
    if wpe_key in ckpt and wpe_key in model_state:
        try:
            ckpt_wpe = ckpt[wpe_key]
            model_wpe = model_state[wpe_key]
            if ckpt_wpe.size() != model_wpe.size():
                min_pos = min(ckpt_wpe.size(0), model_wpe.size(0))
                new_wpe = model_wpe.clone()
                new_wpe[:min_pos] = ckpt_wpe[:min_pos]
                ckpt[wpe_key] = new_wpe
                print(f"Adjusted {wpe_key}: copied first {min_pos} rows from checkpoint")
        except Exception as e:
            ckpt.pop(wpe_key, None)
            print(f"Could not adapt {wpe_key} ({e}); skipping this key")

    try:
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded model from {ckpt_path}")
    except RuntimeError as e:
        print(f"RuntimeError loading model: {e}")
        if wpe_key in str(e) and wpe_key in ckpt:
            ckpt.pop(wpe_key, None)
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded model from {ckpt_path} after removing {wpe_key}")
        else:
            raise
