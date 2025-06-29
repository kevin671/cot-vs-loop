import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


# ---------- グラフ & CPT ----------
def make_random_dag(num_nodes: int, num_edges: int, *, seed: int = 0) -> nx.DiGraph:
    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    possible_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    G.add_edges_from(rng.sample(possible_edges, num_edges))
    assert nx.is_directed_acyclic_graph(G)
    return G


def make_cpts(G: nx.DiGraph, *, seed: int = 0):
    rng = random.Random(seed)
    cpts = {}
    for node in G.nodes:
        parents = list(G.predecessors(node))
        table = {bits: rng.betavariate(1, 1) for bits in itertools.product([0, 1], repeat=len(parents))}
        cpts[node] = (parents, table)
    return cpts


# ---------- 祖先サンプリング ----------
def ancestral_sample(G: nx.DiGraph, cpts) -> Dict[int, int]:
    sample = {}
    for node in nx.topological_sort(G):
        parents, table = cpts[node]
        p1 = table[tuple(sample[p] for p in parents)]
        sample[node] = 1 if random.random() < p1 else 0
    return sample


# ---------- 厳密な条件付き確率 ----------
def compute_exact_conditional_prob(target: int, obs: Dict[int, int], cpts) -> float:
    """P(x_target = 1 | obs) を列挙で正確に計算"""
    vars_all = set(cpts)
    unobserved = list(vars_all - obs.keys() - {target})

    total_p1 = total_all = 0.0
    for bits in itertools.product([0, 1], repeat=len(unobserved)):
        assign = {**obs, **dict(zip(unobserved, bits))}
        # 該当 target の 2 通り
        for t_val in (0, 1):
            assign[target] = t_val
            p = 1.0
            for var, (parents, table) in cpts.items():
                key = tuple(assign[p] for p in parents)
                p_var1 = table[key]
                p *= p_var1 if assign[var] == 1 else (1 - p_var1)
            total_all += p
            if t_val == 1:
                total_p1 += p
    return total_p1 / total_all


def build_train_and_test(
    *,
    num_nodes: int = 15,
    num_edges: int = 25,
    train_samples: int = 50_000,
    # test_samples: int = 500,
    out_dir: str = "bn_dataset_fixed",
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    G = make_random_dag(num_nodes, num_edges, seed=seed)
    cpts = make_cpts(G, seed=seed)
    topo = list(nx.topological_sort(G))
    roots = [v for v in topo if G.in_degree(v) == 0]

    Path(out_dir).mkdir(exist_ok=True)
    train_f = open(Path(out_dir) / "train.txt", "w")
    test_f = open(Path(out_dir) / "test.txt", "w")

    for _ in range(train_samples):
        assign = ancestral_sample(G, cpts)
        seen: List[int] = []
        for q in topo:
            ctx = " ".join(f"{v}= {assign[v]}" for v in seen)
            train_f.write(f"{q}= | {ctx + ' ' if ctx else ''}{q}= {assign[q]}\n")
            seen.append(q)

    # non_roots = [v for v in topo if v not in roots]
    evidence_patterns = list(itertools.product([0, 1], repeat=len(roots)))

    for bits in evidence_patterns:
        evidence = {r: b for r, b in zip(roots, bits)}
        ctx = " ".join(f"{r}= {evidence[r]}" for r in roots)

        # q = rng.choice(non_roots)
        for q in topo:
            p1 = compute_exact_conditional_prob(q, evidence, cpts)
            test_f.write(f"{q}= | {ctx} {q}= {p1:.6f}\n")

    train_f.close()
    test_f.close()


if __name__ == "__main__":
    build_train_and_test(
        num_nodes=15,  # 15,
        num_edges=25,  # 25,
        train_samples=50_000,
        # test_samples=# 1_000,
        out_dir="data/bayes_net",
        seed=42,
    )
