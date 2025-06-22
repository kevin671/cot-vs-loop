import itertools
import os
import random
from collections import deque

import networkx as nx
import numpy as np


def make_random_dag(num_nodes: int, num_edges: int, seed=None):
    nodes = list(range(num_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    possible_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    edges = random.sample(possible_edges, num_edges)
    G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G)
    return G


def make_cpts(G: nx.DiGraph, seed=None):
    cpts = {}
    for node in G.nodes():
        parents = list(G.predecessors(node))
        table = {}
        for bits in itertools.product([0, 1], repeat=len(parents)):
            p = random.betavariate(1, 1)  # 一様ベータ分布
            table[bits] = p
        cpts[node] = (parents, table)
    return cpts


def ancestral_sample(G: nx.DiGraph, cpts):
    sample = {}
    for node in nx.topological_sort(G):
        parents, table = cpts[node]
        key = tuple(sample[p] for p in parents)
        p1 = table[key]
        sample[node] = 1 if random.random() < p1 else 0
    return sample


def neighborhood(G: nx.DiGraph, center: int, radius: int):
    seen = {center}
    q = deque([(center, 0)])
    undirected = G.to_undirected()
    while q:
        u, d = q.popleft()
        if d == radius:
            continue
        for v in undirected.neighbors(u):
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return seen


def generate_dataset(
    G: nx.DiGraph, cpts, num_samples: int, geo_p: float = 0.5, dropout_rate: float = 0.2, max_radius: int = 3, seed=None
):
    data = []
    nodes = list(G.nodes())
    for _ in range(num_samples):
        full = ancestral_sample(G, cpts)
        target = random.choice(nodes)
        k = min(max_radius, np.random.geometric(geo_p))
        neigh = neighborhood(G, target, k)
        obs = {n: full[n] for n in neigh if random.random() > dropout_rate and n != target}
        if not obs:  # 観測ノードが空ならスキップ
            continue
        label = full[target]
        data.append((target, obs, label))
    return data


def save_dataset(data, path="bayes_dataset_custom.txt"):
    with open(path, "w") as f:
        for target, obs, label in data:
            obs_parts = [f"{n} = {v}" for n, v in sorted(obs.items())]
            obs_str = " ".join(obs_parts)
            input_str = f"{target} | {obs_str}"
            output_str = f"{target} = {label}"
            f.write(f"{input_str} {output_str}\n")


def generate_test_dataset(G: nx.DiGraph, cpts, num_samples: int, seed=None):
    data = []
    nodes = list(G.nodes())
    roots = [n for n in nodes if G.in_degree(n) == 0]
    non_roots = [n for n in nodes if G.in_degree(n) > 0]

    for _ in range(num_samples):
        full = ancestral_sample(G, cpts)
        if not roots or not non_roots:
            raise ValueError("Graph must have both root and non-root nodes.")

        obs = {n: full[n] for n in roots}
        target = random.choice(non_roots)
        label = None  # ラベルは使わず、確率計算する

        data.append((target, obs, label))
    return data


def compute_joint_probability(assign, cpts):
    """計算: 指定された変数アサインに対して、joint確率を計算"""
    prob = 1.0
    for var, (parents, table) in cpts.items():
        if var not in assign:
            continue  # varが未指定なら無視
        parent_vals = tuple(assign[p] for p in parents)
        p1 = table[parent_vals]
        if assign[var] == 1:
            prob *= p1
        else:
            prob *= 1 - p1
    return prob


def compute_exact_conditional_prob(target, obs, cpts):
    """計算: P(target=1 | obs)"""
    unobserved_vars = set(cpts.keys()) - set(obs.keys()) - {target}
    unobserved_vars = list(unobserved_vars)

    total_p1 = 0.0
    total_p = 0.0

    for assignment in itertools.product([0, 1], repeat=len(unobserved_vars)):
        full_assign = dict(obs)
        full_assign.update({var: val for var, val in zip(unobserved_vars, assignment)})

        # target=1の確率
        full_assign[target] = 1
        p1 = compute_joint_probability(full_assign, cpts)

        # target=0の確率
        full_assign[target] = 0
        p0 = compute_joint_probability(full_assign, cpts)

        total_p1 += p1
        total_p += p1 + p0

    if total_p == 0:
        return 0.5  # 安全策 (本来ならassertで落とすかも)
    return total_p1 / total_p


def save_dataset_with_prob(data, cpts, path="bayes_dataset_custom.txt"):
    with open(path, "w") as f:
        for target, obs, _ in data:  # _ はサンプルされたラベルは使わない
            prob = compute_exact_conditional_prob(target, obs, cpts)
            obs_parts = [f"{n} = {v}" for n, v in sorted(obs.items())]
            obs_str = " ".join(obs_parts)
            input_str = f"{target} | {obs_str}"
            output_str = f"{target} = {prob:.6f}"
            f.write(f"{input_str} {output_str}\n")


def save_dag(G, path="dag_edges.txt"):
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} -> {v}\n")


def save_cpts(cpts, path="cpts.txt"):
    with open(path, "w") as f:
        for node, (parents, table) in cpts.items():
            f.write(f"Node {node} | Parents: {parents}\n")
            for parent_vals, prob in table.items():
                f.write(f"  {parent_vals} -> {prob:.4f}\n")


def main(
    n_nodes=10,
    n_edges=20,
    n_samples_train=1000000,
    n_samples_test=100,
    geo_p=0.5,
    dropout_rate=0.2,
    max_radius=3,
    seed=42,
    output_dir="data/bayes",
):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "chain"), exist_ok=True)

    G = make_random_dag(n_nodes, n_edges, seed=seed)
    cpts = make_cpts(G, seed=seed + 1)

    save_dag(G, path=os.path.join(output_dir, "dag_edges.txt"))
    save_cpts(cpts, path=os.path.join(output_dir, "cpts.txt"))

    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    print("Generating train dataset...")
    train_data = generate_dataset(G, cpts, n_samples_train, geo_p, dropout_rate, max_radius, seed=seed + 2)
    save_dataset(train_data, path=os.path.join(output_dir, "decoder", "train_data.txt"))
    # save_dataset(train_data, path=os.path.join(output_dir, "chain", "train_data.txt"))

    # 確率を計算する必要がある...
    # 6 | 0 = 0 1 = 0 3 = 1 6 = ? で確率を直接求めるLooped or CoT

    print("Generating test dataset...")
    # test_data = generate_dataset(G, cpts, n_samples_test, geo_p, dropout_rate, max_radius, seed=seed + 3)
    # save_dataset(test_data, path=os.path.join(output_dir, "decoder", "test_data.txt"))
    # save_dataset(test_data, path=os.path.join(output_dir, "chain", "test_data.txt"))
    test_data = generate_test_dataset(G, cpts, n_samples_test, seed=seed + 3)
    save_dataset_with_prob(test_data, cpts, path=os.path.join(output_dir, "decoder", "test_data.txt"))

    print(f"Saved datasets to {output_dir}/train_data.txt and test_data.txt")


if __name__ == "__main__":
    main()
