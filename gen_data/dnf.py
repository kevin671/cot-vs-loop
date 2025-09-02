import random

from tasks.dnf import *


def make_dataset(path: str, n: int, m: int, w: int, seed: int = 123, NUM_EXAMPLES: int = 120):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(NUM_EXAMPLES):
            dnf = gen_random_dnf(n, m, w, rng)
            count = exact_dnf_count(dnf, n)

            for i, conj in enumerate(dnf):
                print(str(i), end=" ", file=f)
                for _, (var_idx, is_neg) in enumerate(conj):
                    print(f"{var_idx}=", end=" ", file=f)
                    print("-1" if is_neg else "+1", end=" ", file=f)

            print("<sep>", end=" ", file=f)

            print(str(count), file=f)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dnf")
    parser.add_argument("--num_vars", type=int, default=NUM_VARS)
    parser.add_argument("--num_clauses", type=int, default=NUM_CLAUSES)
    parser.add_argument("--clause_width", type=int, default=CLAUSE_WIDTH)
    parser.add_argument("--train_size", type=float, default=1e6)
    parser.add_argument("--test_size", type=float, default=1e3)

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Train
    make_dataset(
        path=os.path.join(args.data_dir, "train.txt"),
        n=args.num_vars,
        m=args.num_clauses,
        w=args.clause_width,
        seed=42,
        NUM_EXAMPLES=int(args.train_size),
    )

    # Test
    make_dataset(
        path=os.path.join(args.data_dir, "test.txt"),
        n=args.num_vars,
        m=args.num_clauses,
        w=args.clause_width,
        seed=2024,
        NUM_EXAMPLES=int(args.test_size),
    )
