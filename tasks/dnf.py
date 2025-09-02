import itertools
import math
import random
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from tasks.task import GeneralizationTask

Literal = Tuple[int, bool]  # (var_idx, is_neg)
Conj = List[Literal]
DNF = List[Conj]

NUM_VARS = 10
NUM_CLAUSES = 10
CLAUSE_WIDTH = 3


def gen_random_dnf(n_vars: int, m: int, w: int, rng: random.Random) -> DNF:
    dnf: DNF = []
    for _ in range(m):
        vars_ = rng.sample(range(n_vars), w)  # without replacement
        conj = [(v, rng.random() < 0.5) for v in vars_]
        dnf.append(conj)
    return dnf


def sample_assignment_given_conj(n_vars, rng, conj: Conj):
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


def klm_dnf_count(dnf: DNF, n_vars: int, tau: int = None, rng=None) -> float:
    # epsilon: float = 0.1, delta: float = 0.05, rng=None) -> float:
    if rng is None:
        rng = random.Random()
    m = len(dnf)
    if m == 0:
        return 0.0
    p_clause = [2.0 ** (-len(conj)) for conj in dnf]
    s = sum(p_clause)
    if s == 0.0:
        return 0.0
    # tau = math.ceil(8 * (1 + epsilon) * m * math.log(2 / delta) / (epsilon * epsilon))
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
        # return klm_dnf_count(dnf, n_vars, epsilon / 2, delta, rng)
        return klm_dnf_count(dnf, n_vars, tau * 2, rng)
    mu_hat = (tau * s) / (m * N)
    return mu_hat * (2.0**n_vars)


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


class DNFCountOnlineDataset(IterableDataset):
    def __init__(self, config, split="train", chain=True, seed=42):
        super().__init__()
        self.seed = seed
        self.split = split
        n = config["n"]
        self.n = n  # number of variables
        m = config["m"]
        w = config["w"]
        self.gen_random_dnf = partial(gen_random_dnf, n, m, w)
        self.sample_assignment_given_conj = partial(sample_assignment_given_conj, n)

        dummy_dnf = self.gen_random_dnf(random.Random())
        p_clause = [2.0 ** (-len(conj)) for conj in dummy_dnf]
        self.s = sum(p_clause)
        self.choices = list(range(m))
        self.weights = [pc / self.s for pc in p_clause]

        self.ignore_index = config.get("ignore_index", -100)
        self._build_vocab()

    def _build_vocab(self) -> None:
        self.tok2id: Dict[str, int] = {}
        add = lambda tok: self.tok2id.setdefault(tok, len(self.tok2id))

        add("<pad>")
        add("<sep>")
        add("<eos>")
        add("-1")  # ¬
        add("+1")
        add("Success")
        add("Failure")
        for i in range(self.n):
            add(str(i))
            add(f"{i}=")

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            tokens = []
            dnf = self.gen_random_dnf(rng)
            # add dnf to tokens
            for i, conj in enumerate(dnf):
                tokens.append(str(i))  # index of the clause
                for j, (var_idx, is_neg) in enumerate(conj):
                    tokens.append(f"{var_idx}=")
                    tokens.append("-1" if is_neg else "+1")

            input_dnf_length = len(tokens)

            if self.split == "test":
                gt_count = exact_dnf_count(dnf, self.n)
                yield torch.tensor([self.tok2id[tok] for tok in tokens], dtype=torch.long), torch.tensor(
                    gt_count, dtype=torch.float
                )

            else:
                tokens.append("<sep>")
                j = rng.choices(self.choices, weights=self.weights, k=1)[0]
                tokens.append(str(j))

                tokens.append("<sep>")
                asg = self.sample_assignment_given_conj(rng, dnf[j])
                for i in range(self.n):
                    tokens.append(f"{i}=")
                    tokens.append("+1" if asg[i] else "-1")

                tokens.append("<sep>")
                k = rng.randrange(len(dnf))
                tokens.append(str(k))
                tokens.append("<sep>")
                y = 1 if satisfies(asg, dnf[k]) else 0
                tokens.append("Success" if y == 1 else "Failure")
                tokens.append("<eos>")

                # print(" ".join(tokens))

                input_ids = torch.tensor([self.tok2id[tok] for tok in tokens], dtype=torch.long)

                labels = torch.full_like(input_ids, fill_value=self.ignore_index)
                labels[:-1] = input_ids[1:]
                labels[: input_dnf_length - 1] = self.ignore_index  # ignore the DNF part
                yield input_ids, labels


class DNFCountTask(GeneralizationTask):
    def __init__(self, n, m, w):
        super().__init__()
        self.config = {
            "n": n,  # number of variables
            "m": m,  # number of clauses
            "w": w,  # width of each clause
            "data_dir": "data/dnf",
            "max_length": m * (1 + w * 2) + 3 + 2 * n + 5,
            "ignore_index": -100,
            "vocab_size": 7 + n * 2,
        }

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass  # not used


if __name__ == "__main__":
    n = 10
    m = n // 2
    w = 3

    dnf = gen_random_dnf(n, m, w, random.Random())
    for conj in dnf:
        print(" AND ".join([f"{'' if not is_neg else '-'}x{var_idx}" for var_idx, is_neg in conj]))
    print(dnf)

    # from tasks.dnf import klm_dnf_count

    estimate = klm_dnf_count(dnf, n, tau=100)
    print("Estimate:", estimate)

    exact = exact_dnf_count(dnf, n)
    print("Exact:", exact)

    # task = DNFCountTask(n=20, m=10, w=3)
    # dataset = DNFCountOnlineDataset(task.config, split="test", chain=False, seed=42)
    # for i, (input_ids, y) in enumerate(dataset):
    #    if i >= 10:
    #        break
    #    print(input_ids, y)
