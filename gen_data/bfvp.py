import argparse
import os
import random

nums = ["0", "1"]
binary_signs = ["∧", "∨"]
unary_signs = ["¬"]


def get_expr(x: str):
    if random.random() < 0.2:
        Q = "1" if x == "0" else "0"
        return ["(", "¬", Q, ")"]
    sign = random.choice(binary_signs)
    if sign == "∧":
        if x == "1":
            left, right = "1", "1"
        else:
            left = random.choice(["0", "1"])
            right = "0" if left == "1" else random.choice(["0", "1"])
    else:
        if x == "0":
            left, right = "0", "0"
        else:
            left = random.choice(["0", "1"])
            right = "1" if left == "0" else random.choice(["0", "1"])
    return ["(", left, sign, right, ")"]


def iter_formula(lst: list[str]) -> list[str]:
    while True:
        idx = random.randrange(len(lst))
        if lst[idx] in nums:
            break
    sub = get_expr(lst[idx])
    del lst[idx]
    for tok in reversed(sub):
        lst.insert(idx, tok)
    return lst


def get_sequence(depth: int) -> list[str]:
    lst = [random.choice(nums)]
    history = [lst[:]]
    for _ in range(depth):
        lst = iter_formula(lst)
        history.append(lst[:])
    seq = []
    for stage in reversed(history):
        seq.extend(stage)
        seq.append("=")
    return seq[:-1]


def write_split(path: str, n_examples: int, depth: int, forbid=None) -> set[int]:
    forbid = forbid or set()
    written = 0
    with open(path, "w") as f:
        while written < n_examples:
            seq = tuple(get_sequence(depth))
            h = hash(seq) & 0xFFFFFFFFFFFFFFFF  # 64bit
            if h in forbid:
                continue
            forbid.add(h)

            i_eq = seq.index("=")
            print(" ".join(seq[: i_eq + 1]), seq[-1], file=f)
            written += 1
    return forbid


def build_dataset(depth, train_size, test_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.txt")
    test_path = os.path.join(out_dir, "test.txt")

    hashes = write_split(train_path, train_size, depth)
    write_split(test_path, test_size, depth, forbid=hashes)


if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/bfvp")
    parser.add_argument("--depth", type=int, default=16)  # depth of the formula
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--test_size", type=int, default=10000)
    args = parser.parse_args()
    random.seed(2023)

    data_dir = os.path.join(args.data_dir, str(args.depth))
    build_dataset(args.depth, args.train_size, args.test_size, data_dir)

    # parser.add_argument("--max_depth", type=int, default=64)  # max depth of the formula
    # depths = [2**i for i in range(3, int(math.log2(args.max_depth)) + 1)]
    # for d in depths:
    #    base = os.path.join(args.data_dir, str(d))
    #    build_dataset(d, args.train_size, args.test_size, base)
    #    print(f"Generated data for depth {d} at {base}")
