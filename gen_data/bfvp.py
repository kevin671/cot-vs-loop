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


def build_dataset(depth, train_size, test_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_set, test_set = set(), set()
    while len(train_set) < train_size:
        train_set.add(tuple(get_sequence(depth)))
    while len(test_set) < test_size:
        seq = tuple(get_sequence(depth))
        if seq not in train_set:
            test_set.add(seq)

    def dump(fname, data):
        with open(os.path.join(out_dir, fname), "w") as f:
            for seq in data:
                for tok in seq:
                    print(tok, end=" ", file=f)
                    if tok == "=":
                        break
                print(seq[-1], file=f)

    dump("train.txt", train_set)
    dump("test.txt", test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--test_size", type=int, default=10000)
    args = parser.parse_args()
    random.seed(2023)
    base = os.path.join(args.file, "boolean_formula")
    build_dataset(args.length, args.train_size, args.test_size, base)
    print(f"Written boolean formula-with-negation datasets to {base}")
