import itertools
import os
import random
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from tasks.task import CurriculumDataset, GeneralizationTask


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


def ancestral_sample(G: nx.DiGraph, cpts) -> Dict[int, int]:
    sample = {}
    for node in nx.topological_sort(G):
        parents, table = cpts[node]
        p1 = table[tuple(sample[p] for p in parents)]
        sample[node] = 1 if random.random() < p1 else 0
    return sample


class BayesNetOnlineDataset(torch.utils.data.IterableDataset):
    def __init__(self, config, split: str = "train", seed: int = 42):
        super().__init__()
        self.num_nodes = config["num_nodes"]
        self.num_edges = config["num_edges"]
        self.max_len = config["max_length"]
        self.ignore_index = config["ignore_index"]

        self.G = make_random_dag(self.num_nodes, self.num_edges, seed)
        self.cpts = make_cpts(self.G, seed)
        self.topo = list(nx.topological_sort(self.G))

        self._build_vocab()

    def _build_vocab(self) -> None:
        self.tok2id: Dict[str, int] = {}
        add = lambda tok: self.tok2id.setdefault(tok, len(self.tok2id))

        add("<pad>")
        add("|")
        add("0")
        add("1")
        for i in range(self.num_nodes):
            add(f"{i}=")

        self.pad_id = self.tok2id["<pad>"]

    def _encode_tokens(self, toks: List[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]

    def __iter__(self):
        rng = random.Random(torch.initial_seed())
        while True:
            assign = ancestral_sample(self.G, self.cpts)

            seen = []
            for q in self.topo:
                ctx_tokens = [f"{v}= {assign[v]}" for v in seen]
                sample_tokens = [f"{q}=", "|", *ctx_tokens, f"{q}= {assign[q]}"]
                ids = [self.tok2id[t] for t in sample_tokens]

                pad = [self.tok2id["<pad>"]] * (self.max_len - len(ids))
                input_ids = ids + pad

                labels = torch.full((self.max_len,), self.ignore_index, dtype=torch.long)
                labels[len(ids) - 1] = self.tok2id[str(assign[q])]  # 0/1 トークンの id

                yield torch.tensor(input_ids), labels
                seen.append(q)


class BayesNetTask(GeneralizationTask):
    # Warning: Use causal mask for this task
    num_nodes = 10
    config = {
        "name": "bayes_net",
        "description": "Ancestor sampling in Bayesian networks.",
        "data_dir": "data/bayes_net",
        "num_nodes": num_nodes,
        "num_edges": num_nodes * 2,
        "max_length": num_nodes * 2 + 2,
        "ignore_index": -100,
    }
    config["vocab_size"] = num_nodes + 4

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=self.config["ignore_index"],
        )

    def _select_logits01(self, last_logits: torch.Tensor) -> torch.Tensor:
        ds = getattr(self, "_cached_ds", None)
        if ds is None:
            # TODO: CoT用の評価へ
            self._cached_ds = None  # BayesNetDataset(self.config, split="test")
            ds = self._cached_ds
        logits01 = torch.stack((last_logits[:, ds.id0], last_logits[:, ds.id1]), dim=-1)  # (B,2)
        return logits01

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # we assume the test data has the same length, so we can use the last time step
        # this is for looped
        last_logits = output[:, -1, :]
        logits01 = self._select_logits01(last_logits)
        prob1 = F.softmax(logits01, dim=-1)[:, 1]
        rmse = torch.sqrt(F.mse_loss(prob1, target.float()))
        return rmse


if __name__ == "__main__":
    # Example usage
    task = BayesNetTask()
    dataset = BayesNetOnlineDataset(task.config, split="test")
    print(f"Number of samples in {dataset.split} set: {len(dataset)}")
    for i in range(5):
        input_ids, label = dataset[i]
        print(f"Input IDs: {input_ids}, Label: {label}")
