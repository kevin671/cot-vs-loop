# https://github.com/guyuntian/CoT_benchmark/blob/main/ED/data.py
import argparse
import os

import numpy as np

parser = argparse.ArgumentParser(description="data")

parser.add_argument("--data_dir", type=str, default="Data")
parser.add_argument("--length", type=int, default=32)
parser.add_argument("--train_size", type=int, default=100000)
parser.add_argument("--test_size", type=int, default=10000)
parser.add_argument("--using", type=int, default=8)
# The costs for insertion, deletion, replacement, and matching.
parser.add_argument("--insert_cost", type=int, default=2)
parser.add_argument("--delete_cost", type=int, default=2)
parser.add_argument("--replace_cost", type=int, default=3)
parser.add_argument("--match_cost", type=int, default=0)
parser.add_argument("--objective", type=str, default="min", choices=["max", "min"])
parser.add_argument("--make_chain", action="store_true", default=False)

args = parser.parse_args()

np.random.seed(2023)

alphabet = [i for i in "abcdefghijklmnopqrstuvwxyz"]

max_len = 0
max_history = 0


def get_seq(diff):
    using = np.random.randint(args.using) + 3
    available = np.random.choice(alphabet, using, replace=False)
    str1 = np.random.randint(using, size=args.length)
    str1 = [available[i] for i in str1]
    if np.random.rand() < 0.4:
        length = np.random.randint(args.length - 3, args.length + 3)
        str2 = np.random.randint(using, size=length)
        str2 = [available[i] for i in str2]
    else:
        str2 = str1[:]
        for _ in range(diff):
            a = np.random.randint(3)
            if a == 0 and len(str2) > 2:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + str2[p + 1 :]
            elif a == 1:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + [np.random.choice(available)] + str2[p + 1 :]
            else:
                p = np.random.randint(len(str2) + 1)
                str2 = str2[:p] + [np.random.choice(available)] + str2[p:]
    if str1 == str2 or len(str2) >= args.length + 3 or len(str2) < args.length - 3:
        return get_seq(diff)
    if len(str1) > len(str2):
        return str2, str1
    return str1, str2


def solve(str1, str2):
    insert_cost = args.insert_cost
    delete_cost = args.delete_cost
    replace_cost = args.replace_cost
    match_cost = args.match_cost
    objective = args.objective

    opt_fn = min if objective == "min" else max

    m, n = len(str1), len(str2)
    matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        matrix[i][0] = matrix[i - 1][0] + delete_cost
    for j in range(1, n + 1):
        matrix[0][j] = matrix[0][j - 1] + insert_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = match_cost
            else:
                cost = replace_cost
            matrix[i][j] = opt_fn(
                matrix[i - 1][j] + delete_cost,  # Deletion
                matrix[i][j - 1] + insert_cost,  # Insertion
                matrix[i - 1][j - 1] + cost,
            )

    return matrix if args.make_chain else matrix[-1][-1]


def write_train(path, n_examples, args):
    global max_history, max_len
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hashes = set()
    with open(path, "w") as f:
        for _ in range(n_examples):
            str1, str2 = get_seq(np.random.randint(args.length) + 1)
            matrix = solve(str1, str2)
            final = str1 + ["|"] + str2 + ["<sep>"]

            # 64-bit hash だけ覚えておけばメモリ << 100 MB
            h = hash(tuple(map(str, final))) & 0xFFFFFFFFFFFFFFFF
            hashes.add(h)

            for row in range(1, len(matrix)):
                final = final + matrix[row][1:] + [","]
            final.append("<sep>")
            final.append(matrix[-1][-1])
            max_history = max(max_history, len(final))
            max_len = max(max_len, len(str1) + len(str2) + 3)

            print(" ".join(map(str, final)), file=f)
    return hashes


def write_test(path, n_examples, args, forbid):
    global max_history, max_len
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = 0
    with open(path, "w") as f:
        while written < n_examples:
            str1, str2 = get_seq(np.random.randint(args.length) + 1)
            matrix = solve(str1, str2)
            final = str1 + ["|"] + str2 + ["<sep>"]

            h = hash(tuple(map(str, final))) & 0xFFFFFFFFFFFFFFFF
            if h in forbid:  # train か既出 test と衝突
                continue
            forbid.add(h)

            for row in range(1, len(matrix)):
                final = final + matrix[row][1:] + [","]
            final.append("<sep>")
            final.append(matrix[-1][-1])
            max_history = max(max_history, len(final))
            max_len = max(max_len, len(str1) + len(str2) + 3)

            print(" ".join(map(str, final)), file=f)

            written += 1


data_dir = f"{args.data_dir}/{args.length}"
if args.make_chain:
    costs = dict(insert=2, delete=2, replace=3, match=0, chain=False)

    out_dir = os.path.join(args.data_dir, str(args.length))
    train_path = os.path.join(out_dir, "chain/train_data.txt")
    test_path = os.path.join(out_dir, "chain/test_data.txt")

    hashes = write_train(train_path, args.train_size, args)
    write_test(test_path, args.test_size, args, hashes)

    print(f"max direct len:{max_len}")
    print(f"max history len:{max_history}")

else:
    train_set = set()
    max_len = 0

    while len(train_set) < args.train_size:
        str1, str2 = get_seq(np.random.randint(args.length) + 1)
        final = str1 + ["|"] + str2 + ["<sep>"]
        final.append(solve(str1, str2))
        train_set.add(tuple(final))
        max_len = max(max_len, len(str1) + len(str2) + 3)

    test_set = set()
    while len(test_set) < args.test_size:
        str1, str2 = get_seq(np.random.randint(args.length) + 1)
        final = str1 + ["|"] + str2 + ["<sep>"]
        final.append(solve(str1, str2))
        if tuple(final) not in train_set:
            test_set.add(tuple(final))
        max_len = max(max_len, len(str1) + len(str2) + 3)

    os.makedirs(data_dir, exist_ok=True)
    decoder = f"{data_dir}/decoder"
    os.makedirs(decoder, exist_ok=True)

    with open(f"{decoder}/train_data.txt", "w") as f1:
        for lst in train_set:
            for i in lst:
                print(i, end=" ", file=f1)
                if i == "<sep>":
                    break
            print(lst[-1], file=f1)

    with open(f"{decoder}/test_data.txt", "w") as f1:
        for lst in test_set:
            for i in lst:
                print(i, end=" ", file=f1)
                if i == "<sep>":
                    break
            print(lst[-1], file=f1)

    print(f"max direct len:{max_len}")
