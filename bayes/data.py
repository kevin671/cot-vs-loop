import itertools
import os
import random
from collections import deque

import networkx as nx
import numpy as np


def make_random_dag(num_nodes: int, num_edges: int, seed=None):
    random.seed(seed)
    nodes = list(range(num_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    possible_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    edges = random.sample(possible_edges, num_edges)
    G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G)
    return G


def make_cpts(G: nx.DiGraph, seed=None):
    random.seed(seed)
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
    random.seed(seed)
    np.random.seed(seed)
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


def main(
    n_nodes=10,
    n_edges=20,
    n_samples_train=1000000,
    n_samples_test=100000,
    geo_p=0.5,
    dropout_rate=0.2,
    max_radius=3,
    seed=42,
    output_dir="data/bayes",
):
    os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "chain"), exist_ok=True)

    G = make_random_dag(n_nodes, n_edges, seed=seed)
    cpts = make_cpts(G, seed=seed + 1)

    print("Generating train dataset...")
    train_data = generate_dataset(G, cpts, n_samples_train, geo_p, dropout_rate, max_radius, seed=seed + 2)
    save_dataset(train_data, path=os.path.join(output_dir, "decoder", "train_data.txt"))
    # save_dataset(train_data, path=os.path.join(output_dir, "chain", "train_data.txt"))

    print("Generating test dataset...")
    test_data = generate_dataset(G, cpts, n_samples_test, geo_p, dropout_rate, max_radius, seed=seed + 3)
    save_dataset(test_data, path=os.path.join(output_dir, "decoder", "test_data.txt"))
    # save_dataset(test_data, path=os.path.join(output_dir, "chain", "test_data.txt"))

    print(f"Saved datasets to {output_dir}/train_data.txt and test_data.txt")


if __name__ == "__main__":
    main()
