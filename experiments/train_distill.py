import argparse
import math
import os

import torch
from tqdm import tqdm
from transformers import set_seed

import wandb
from models import build_model
from tasks import get_task_and_datasets

from .utils import load_checkpoint_adapt_wpe, set_optimizer_scheduler


# https://github.com/da03/Internalize_CoT_Step_by_Step/blob/main/src/train.py#L25
def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    if removal_smoothing_lambda == float("inf"):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1 - cum_prob)
    return lambda_distribution


def main():
    parser = argparse.ArgumentParser(description="train")
    # Dataset parameters
    parser.add_argument("--task", type=str, default="arithmetic")
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--min_input_size", type=int, default=None)
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Model parameters
    parser.add_argument("--model", type=str, default="Looped", choices=["Looped", "GPT", "TMLT", "CT"])
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_loop", type=int, default=16)
    parser.add_argument("--n_step", type=int, default=4)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--use_rope", action="store_true")
    # Else
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--min_cot_length", type=int, default=0)
    parser.add_argument("--max_cot_length", type=int, default=None)
    parser.add_argument("--epochs_per_stage", type=int, default=1)
    parser.add_argument("--remove_per_stage", type=int, default=2)
    parser.add_argument("--removal_smoothing_lambda", type=float, default=4)
    args = parser.parse_args()
    print(args)

    seed = 42
    set_seed(seed)
    os.makedirs("./output", exist_ok=True)
    is_coconut = args.model == "CT"

    lambda_distribution = compute_lambda_distribution(args.removal_smoothing_lambda)
    print(lambda_distribution.tolist()[:10])

    task, train_dataset, test_dataset = get_task_and_datasets(
        args,
        chain=args.chain,
        cot_length=args.max_cot_length,
        is_coconut=is_coconut,
        lambda_distribution=lambda_distribution,  # new
    )
    cur_cot_length = task.config["max_length"] if args.max_cot_length is None else args.max_cot_length
    n_latent_steps = 1  # args.n_step
    print(task.config)

    collate_fn = getattr(task, "collate_fn", None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    if args.chain and args.task != "word":
        test_batch_size = 1
        test_dataset = torch.utils.data.Subset(test_dataset, range(100))  # 1000
    else:
        test_batch_size = args.batch_size
        test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = build_model(args, task)
    model.cuda()

    if args.model_path:
        load_checkpoint_adapt_wpe(model, args.model_path)

    optimizer, scheduler = set_optimizer_scheduler(model, args, train_loader)

    wandb.init(project="cotloop_distill_2", config=args, name=f"{args.task}_{args.input_size}_{args.model}")

    for epoch in range(args.epoch):
        print(f"Epoch {epoch + 1}/{args.epoch}")
        model.train()

        if epoch > 0 and epoch % args.epochs_per_stage == 0:
            # if epoch > 0 and (epoch == 1 or epoch % args.epochs_per_stage == 0):
            cur_cot_length = max(args.min_cot_length, cur_cot_length - args.remove_per_stage)
            task, train_dataset, test_dataset = get_task_and_datasets(
                args,
                chain=args.chain,
                cot_length=cur_cot_length,
                is_coconut=is_coconut,
                lambda_distribution=lambda_distribution,  # new
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
            )
            if args.chain and args.task != "word":
                test_batch_size = 1
                test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
            else:
                test_batch_size = args.batch_size
                test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn
            )

            del optimizer
            optimizer, scheduler = set_optimizer_scheduler(model, args, train_loader)

            n_latent_steps = min(n_latent_steps + 1, args.n_step)

        print(f"Current CoT length: {cur_cot_length}")
        wandb.log({"cur_cot_length": cur_cot_length})

        for i, batch in enumerate(tqdm(train_loader)):
            if is_coconut:
                input_ids, cot_ids, y = batch
                inputs, cot_inputs, y = (
                    input_ids.cuda(),
                    cot_ids.cuda(),
                    y.long().cuda(),
                )
                logits = model(inputs, cot_inputs, n_latent_steps=n_latent_steps)
            else:
                input_ids, y = batch
                inputs, y = input_ids.cuda(), y.long().cuda()
                logits = model(inputs)

            loss = task.pointwise_loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": loss.item(), "lr": lr})

        if (epoch + 1) % args.val_interval == 0:
            # Save the model
            out_dir = os.path.join(args.output_dir, wandb.run.id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/latest.pt")

            # Evaluate on the test set
            model.eval()
            seq_len = args.input_size
            print(f"Evaluating on test set with sequence length: {seq_len}")
            with torch.no_grad():
                if args.task == "word" and args.chain is False:
                    total_acc = torch.zeros(task.config["max_length"], device="cpu")
                else:
                    total_acc = torch.tensor(0.0, device="cpu")
                for i, batch in enumerate(tqdm(test_loader)):
                    # unpack batch depending on whether model uses continuous-thought (CT)
                    if is_coconut:
                        # input_ids, _, y = batch
                        input_ids, y = batch
                        inputs, y = input_ids.cuda(), y.long().cuda()
                    else:
                        input_ids, y = batch
                        inputs, y = input_ids.cuda(), y.long().cuda()

                    if args.chain:
                        if args.task == "word":
                            max_new_tokens = cur_cot_length + 2
                        else:
                            max_new_tokens = (
                                args.input_size * 3 + cur_cot_length if cur_cot_length else task.config["max_length"]
                            )

                        # For generation we pass only the prompt tokens (`inputs`).
                        if is_coconut:
                            idx = model.generate(
                                inputs,
                                top_k=1,
                                max_new_tokens=max_new_tokens,
                                n_latent_steps=n_latent_steps,
                            )
                        else:
                            idx = model.generate(inputs, top_k=1, max_new_tokens=max_new_tokens)
                        acc = task.accuracy_fn(idx, y).detach().cpu()
                        print(f"Accuracy at step {i}: {acc.item()}", flush=True)
                    else:  # Looped
                        logits = model(inputs)
                        acc = task.accuracy_fn(logits, y).detach().cpu()
                    total_acc += acc
            avg_acc = total_acc / len(test_loader)

            wandb.log({"test_accuracy": avg_acc})
            print(f"Epoch {epoch + 1}, Test Accuracy: {avg_acc}")


if __name__ == "__main__":
    main()
