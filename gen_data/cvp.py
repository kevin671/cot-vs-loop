# cvp_dataset.py
import random
from typing import List, Tuple

GATE_TYPES = ["AND", "OR", "NOT"]
CONST_TYPES = ["TRUE", "FALSE"]


def _gate_value(gtype: str, a: int, b: int) -> int:
    if gtype == "TRUE":
        return 1
    if gtype == "FALSE":
        return 0
    if gtype == "NOT":
        return 1 - a
    if gtype == "AND":
        return a & b
    if gtype == "OR":
        return a | b
    raise ValueError(f"Unknown gate type {gtype}")


def generate_circuit(
    m: int, p_const: float = 0.2, p_not: float = 0.2, seed: int | None = None
) -> Tuple[List[str], int]:
    """
    1 ≦ m : ゲート数（トポロジカル順に 1…m）
    戻り値:
        tokens … 生成されたトークン列 (末尾に '=' を含む)
        label  … 最終ゲート m の値 (0/1)
    """
    if seed is not None:
        random.seed(seed)

    gate_vals: List[int] = [None] * (m + 1)  # 1-indexed
    tokens: list[str] = []

    for gid in range(1, m + 1):
        # ゲートタイプを決定
        if gid <= 2 or random.random() < p_const:  # 最初は必ず定数にして安定化
            gtype = random.choice(CONST_TYPES)
        else:
            if random.random() < p_not:
                gtype = "NOT"
            else:
                gtype = random.choice(["AND", "OR"])

        # 入力ゲートと値の決定
        if gtype in CONST_TYPES:
            in1_id = in2_id = "NA"
            val = _gate_value(gtype, 0, 0)
        elif gtype == "NOT":
            src = random.randrange(1, gid)  # 既に確定しているゲート
            in1_id, in2_id = src, "NA"
            val = _gate_value(gtype, gate_vals[src], 0)
        else:  # AND / OR
            a = random.randrange(1, gid)
            b = random.randrange(1, gid)
            in1_id, in2_id = a, b
            val = _gate_value(gtype, gate_vals[a], gate_vals[b])

        gate_vals[gid] = val
        tokens.extend([gtype, str(in1_id), str(in2_id), str(gid)])

    tokens.append("=")
    return tokens, gate_vals[m]


def write_dataset(n_samples: int, m_min: int, m_max: int, path: str, seed: int | None = None) -> None:
    """
    `path` にタブ区切りで
        <input_tokens_str>\t<label>
    を n_samples 行書き出す．
    """
    if seed is not None:
        random.seed(seed)

    with open(path, "w") as f:
        for _ in range(n_samples):
            m = random.randint(m_min, m_max)
            tokens, label = generate_circuit(m)
            line = " ".join(tokens) + f"\t{label}\n"
            f.write(line)


if __name__ == "__main__":
    # 例: トレーニング用 1e6 サンプル, テスト用 1e4 サンプル
    write_dataset(1_0, 4, 32, "cvp_train.txt", seed=42)
    write_dataset(10, 4, 32, "cvp_test.txt", seed=123)
