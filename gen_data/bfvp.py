import argparse
import os
import random

parser = argparse.ArgumentParser(description="boolean formula dataset with negation")
parser.add_argument("--file", type=str, default="data")
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--train_size", type=int, default=100000)
parser.add_argument("--test_size", type=int, default=10000)
args = parser.parse_args()

# 定数は "0","1"
nums = ["0", "1"]
random.seed(2023)

# 二項演算子と単項演算子を区別
binary_signs = ["∧", "∨"]
unary_signs = ["¬"]


def get_expr(x: str):
    """
    評価結果 x になる部分式を返す。
    - 二項演算: (Q op R) が x になるよう Q,R を逆生成
    - 単項演算: ¬Q が x になるよう Q を逆生成
    """
    # まず二項・単項のどちらにするか決定（ここでは 20% で否定を使う例）
    if random.random() < 0.2:
        # --- 単項 negation ---
        # x = ¬Q のとき Q = ¬x
        Q = "1" if x == "0" else "0"
        # 外側でカッコをつけてもつけなくてもOK
        return ["(", "¬", Q, ")"]
    else:
        # --- 二項演算 ---
        sign = random.choice(binary_signs)
        if sign == "∧":
            if x == "1":
                left, right = "1", "1"
            else:  # x == "0"
                # (0,0),(0,1),(1,0) のいずれか
                left = random.choice(["0", "1"])
                right = "0" if left == "1" else random.choice(["0", "1"])
        else:  # sign == "∨"
            if x == "0":
                left, right = "0", "0"
            else:  # x == "1"
                # (1,0),(0,1),(1,1) のいずれか
                left = random.choice(["0", "1"])
                right = "1" if left == "0" else random.choice(["0", "1"])
        return ["(", left, sign, right, ")"]


def iter_formula(lst: list[str]) -> list[str]:
    # lst 中の定数またはサブ式の先頭 "(" を狙って置き換え可能にしてもいいのですが、
    # ここではシンプルに「0か1」を選んで、そのトークンを丸ごと置き換えます。
    while True:
        idx = random.randrange(len(lst))
        if lst[idx] in nums:
            break

    orig = lst[idx]
    sub = get_expr(orig)

    # 置き換え
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

    # 各ステージを逆順で "=" つなぎ
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
    base = os.path.join(args.file, "boolean_formula")
    build_dataset(args.length, args.train_size, args.test_size, base)
    print(f"Written boolean formula-with-negation datasets to {base}")
