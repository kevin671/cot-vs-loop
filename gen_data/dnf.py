import itertools
import json
import math
import random
from typing import List, Tuple

Literal = Tuple[int, bool]  # (var_idx, is_neg)
Conj = List[Literal]
DNF = List[Conj]


def klm_dnf_count(dnf: DNF, n_vars: int, epsilon: float = 0.1, delta: float = 0.05, rng=None) -> float:
    if rng is None:
        rng = random.Random()
    m = len(dnf)
    if m == 0:
        return 0.0
    p_clause = [2.0 ** (-len(conj)) for conj in dnf]
    s = sum(p_clause)
    if s == 0.0:
        return 0.0
    tau = math.ceil(8 * (1 + epsilon) * m * math.log(2 / delta) / (epsilon * epsilon))
    choices = list(range(m))
    weights = [pc / s for pc in p_clause]

    def sample_assignment_given_conj(conj: Conj):
        asg = [False] * n_vars
        fixed = [False] * n_vars
        for i, is_neg in conj:
            asg[i] = not is_neg
            fixed[i] = True
        for i in range(n_vars):
            if not fixed[i]:
                asg[i] = rng.random() < 0.5
        return asg

    def satisfies(asg, conj):
        return all((not is_neg) == asg[i] for (i, is_neg) in conj)

    N = 0
    for _ in range(tau):
        j = rng.choices(choices, weights=weights, k=1)[0]
        asg = sample_assignment_given_conj(dnf[j])
        k = rng.randrange(m)
        if satisfies(asg, dnf[k]):
            N += 1
    if N == 0:
        # very rare; retry with relaxed eps
        return klm_dnf_count(dnf, n_vars, epsilon / 2, delta, rng)
    mu_hat = (tau * s) / (m * N)
    return mu_hat * (2.0**n_vars)


# ==== Exact brute-force (for small n) ====
def exact_dnf_count(dnf: DNF, n_vars: int) -> int:
    total = 0
    for bits in itertools.product([False, True], repeat=n_vars):
        sat = False
        for conj in dnf:
            ok = True
            for i, is_neg in conj:
                if (not is_neg) != bits[i]:
                    ok = False
                    break
            if ok:
                sat = True
                break
        if sat:
            total += 1
    return total


def gen_random_dnf(n_vars: int, m: int, w: int, rng: random.Random) -> DNF:
    dnf: DNF = []
    for _ in range(m):
        vars_ = rng.sample(range(n_vars), w)  # without replacement
        conj = [(v, rng.random() < 0.5) for v in vars_]
        dnf.append(conj)
    return dnf


def dnf_to_string(dnf: DNF) -> str:
    # Pretty printable DNF for prompts: (x0 & ~x1 & x5) | (x2) | ...
    def lit_str(l):
        i, neg = l
        return f"~x{i}" if neg else f"x{i}"

    conj_strs = ["(" + " & ".join(lit_str(l) for l in conj) + ")" for conj in dnf]
    return " | ".join(conj_strs) if conj_strs else "()"


def make_dataset(path: str, seed: int = 123, NUM_EXAMPLES: int = 120):
    rng = random.Random(seed)
    # Distributions over (n, m, w): m = alpha*n, w ~ {2,3,4}
    # ns = [10, 12, 14, 16]  # small enough for exact; you can add larger n later
    # alphas = [0.25, 0.5, 0.75]
    # ws = [2, 3, 4]
    n = 10
    w = 3
    m = int(n * 0.5)
    # with open(path, "w", encoding="utf-8") as f:
    for idx in range(NUM_EXAMPLES):
        # n = rng.choice(ns)
        # m = int(n * rng.choice(alphas))
        # w_sampler = lambda: rng.choice(ws)
        dnf = gen_random_dnf(n, m, w, rng)
        # compute label
        if n <= 16:
            count = exact_dnf_count(dnf, n)
        else:
            count = klm_dnf_count(dnf, n, epsilon=0.1, delta=0.05, rng=rng)

        dnf_str = dnf_to_string(dnf)

    return path


out_path = "/mnt/data/dnf_count_dataset_demo_2025-09-02.jsonl"
p = make_dataset(out_path, NUM_EXAMPLES=120)
