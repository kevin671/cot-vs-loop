# https://github.com/guyuntian/CoT_benchmark/blob/main/arithmetic/data.py
import argparse
import os
import random

import numpy as np

parser = argparse.ArgumentParser(description="data")

parser.add_argument("--file", type=str, default="data")
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--train_size", type=float, default=1e6)
parser.add_argument("--test_size", type=float, default=1e5)
parser.add_argument("--number_range", type=int, default=11)
parser.add_argument("--under", action="store_true", default=False)

args = parser.parse_args()
np.random.seed(2023)

rang = args.number_range
nums = [i for i in range(rang)]
mul = np.arange(rang).reshape(-1, 1) * np.arange(rang).reshape(1, -1)
mul = mul % rang

div = np.zeros((rang, rang), dtype=np.int32)
for i in range(rang):
    pos = np.where(mul == i)
    div[i, pos[0]] = pos[1]


def operator(a, b, rule):
    if rule == "+":
        x = (a + b) % rang
        return x
    if rule == "-":
        x = (a - b) % rang
        return x
    if rule == "*":
        x = mul[a, b]
        return x
    if rule == "/":
        x = div[a, b]
        return x


signs = ["+", "-", "*", "/"]
random.seed(2023)


def get_expr(x):
    while True:
        num1 = random.choice(nums)
        sign = random.choice(signs)
        if sign == "+":
            num2 = operator(x, num1, "-")
        elif sign == "-":
            num2 = operator(num1, x, "-")
        elif sign == "*":
            if num1 == 0 and x != 0:
                continue
            elif num1 == 0:
                num2 = random.choice(nums)
            else:
                num2 = operator(x, num1, "/")
        else:
            if x == 0 and num1 == 0:
                num2 = random.choice(nums)
                if num2 == 0:
                    continue
            elif x == 0 or num1 == 0:
                continue
            else:
                num2 = operator(num1, x, "/")
        break
    return ["(", str(num1), sign, str(num2), ")"]


def iter(lst):
    while True:
        item = random.randint(0, len(lst) - 1)
        if lst[item].isdigit():
            break
    expr = get_expr(int(lst[item]))
    need_bracket = False
    if item > 0 and lst[item - 1] == "/":
        need_bracket = True
    elif (item > 0 and lst[item - 1] == "*") or (item < (len(lst) - 1) and lst[item + 1] in ["*", "/"]):
        if expr[2] in ["+", "-"]:
            need_bracket = True
    elif item > 0 and lst[item - 1] == "-":
        if expr[2] in ["+", "-"]:
            need_bracket = True
    del lst[item]
    if not need_bracket:
        expr = expr[1:-1]
    for i in reversed(expr):
        lst.insert(item, i)
    return lst


def get(length):
    lst = [str(random.choice(nums))]
    history = [lst[:]]
    for _ in range(length):
        lst = iter(lst)
        history.append(lst[:])
    ans = []
    for item in reversed(history):
        ans = ans + item + ["="]
    return ans[:-1]


def build_dataset(depth: int, train_size: int, test_size: int, out_dir: str, fname: str = "arithmetic") -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_set, test_set = set(), set()

    # Train
    while len(train_set) < train_size:
        train_set.add(tuple(get(depth)))
    # Test
    while len(test_set) < test_size:
        h = tuple(get(depth))
        if h not in train_set:
            test_set.add(h)

    # ---- 書き込み ("最終式 = 答え" だけ) ----
    def dump(fname, data):
        with open(os.path.join(out_dir, fname), "w") as f:
            for hist in data:
                for tok in hist:  # 最終式～最初の '=' まで出力
                    print(tok, end=" ", file=f)
                    if tok == "=":
                        break
                print(hist[-1], file=f)  # ラベル（最終値）

    dump("train.txt", train_set)
    dump("test.txt", test_set)


if __name__ == "__main__":
    base_dir = f"{args.file}/arithmetic"

    if args.under:
        d = 2
        while d <= args.length:
            subdir = os.path.join(base_dir, f"len_{d}")
            build_dataset(d, int(args.train_size), int(args.test_size), subdir)
            d *= 2
            print(f"Dataset for length {d} written to: {subdir}")
    else:
        build_dataset(args.length, int(args.train_size), int(args.test_size), base_dir)

    print(f"Datasets successfully written to: {base_dir}")
