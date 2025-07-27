import itertools
import random
from typing import Dict, List

import networkx as nx
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from tasks.task import GeneralizationTask


def make_random_dag(num_nodes: int, num_edges: int, seed: int = 0) -> nx.DiGraph:
    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    possible_edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
    G.add_edges_from(rng.sample(possible_edges, num_edges))
    assert nx.is_directed_acyclic_graph(G)
    return G


def make_cpts(G: nx.DiGraph, seed: int = 0):
    rng = random.Random(seed)
    cpts = {}
    for node in G.nodes:
        parents = list(G.predecessors(node))
        table = {bits: rng.betavariate(1, 1) for bits in itertools.product([0, 1], repeat=len(parents))}
        cpts[node] = (parents, table)
    return cpts


def ancestral_sample(G: nx.DiGraph, cpts, rng: random.Random) -> Dict[int, int]:
    sample = {}
    for node in nx.topological_sort(G):
        parents, table = cpts[node]
        p1 = table[tuple(sample[p] for p in parents)]
        sample[node] = 1 if rng.random() < p1 else 0
    return sample


def deterministic_cpts(G: nx.DiGraph, seed: int = 0):
    rng = random.Random(seed)
    cpts = {}

    for node in G.nodes:
        parents = list(G.predecessors(node))
        arity = len(parents)
        flip = tuple(rng.randint(0, 1) for _ in range(arity))

        if arity == 0:
            table = {(): 0.5}
        else:
            table = {
                bits: sum(b ^ f for b, f in zip(bits, flip)) % 2 for bits in itertools.product([0, 1], repeat=arity)
            }

        cpts[node] = (parents, table)

    return cpts


def ancestral_sample_det(G: nx.DiGraph, cpts, rng: random.Random) -> Dict[int, int]:
    sample = {}
    for node in nx.topological_sort(G):
        parents, table = cpts[node]
        if len(parents) == 0:
            sample[node] = 1 if rng.random() < 0.5 else 0
        else:
            sample[node] = table[tuple(sample[p] for p in parents)]
    return sample


class BayesNetOnlineDataset(IterableDataset):
    def __init__(self, config, deterministic=False, split: str = "train", seed: int = 42, chain: bool = False):
        super().__init__()
        self.num_nodes = config["num_nodes"]
        self.num_edges = config["num_edges"]
        self.max_len = config["max_length"]
        self.ignore_index = config["ignore_index"]
        self.mode = split

        self.G = make_random_dag(self.num_nodes, self.num_edges, seed)
        if deterministic:
            self.cpts = deterministic_cpts(self.G)
            self._sample_fn = ancestral_sample_det
        else:
            self.cpts = make_cpts(self.G, seed)
            self._sample_fn = ancestral_sample

        self.topo = list(nx.topological_sort(self.G))

        self.roots = [v for v in self.G.nodes if self.G.in_degree(v) == 0]
        self.non_root_indices = [i for i, v in enumerate(self.topo) if v not in self.roots]
        self.query_node = self.topo[-1]
        self.chain = chain

        self._build_vocab()
        # print(self.cpts)

    def _build_vocab(self) -> None:
        self.tok2id: Dict[str, int] = {}
        add = lambda tok: self.tok2id.setdefault(tok, len(self.tok2id))

        add("<pad>")
        add("<mask>")
        add("<eos>")
        add("0")
        add("1")
        for i in range(self.num_nodes):
            add(f"{i}=")

        self.pad_id = self.tok2id["<pad>"]
        # self.mask_id = self.tok2id["<mask>"]

    def _encode_tokens(self, toks: List[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]

    def __iter__(self):
        rng = random.Random(torch.initial_seed() % (2**32))
        while True:
            assign = self._sample_fn(self.G, self.cpts, rng)
            if self.mode == "train":
                if self.chain:
                    # q_idx = rng.choice(self.non_root_indices)
                    # q = self.topo[q_idx]
                    # ctx = self.topo[: q_idx + 1]

                    tokens, labels = [], []

                    for node in self.topo:
                        var_tok = f"{node}="
                        val_tok = str(assign[node])
                        tokens.extend([var_tok, val_tok])
                        # labels.extend([self.tok2id[val_tok], self.ignore_index])
                else:
                    # MASK_PROB = 0.15
                    MASK_TOKEN = "<mask>"
                    q_idx = rng.choice(self.non_root_indices)
                    q = self.topo[q_idx]
                    pa_q = set(self.G.predecessors(q))

                    tokens: list[str] = []
                    labels: list[int] = []

                    for node in self.topo:
                        var_tok = f"{node}="
                        if node == q:
                            tokens.extend([var_tok, MASK_TOKEN])
                            labels.extend([self.ignore_index, self.tok2id[str(assign[node])]])

                        elif (node in self.roots) or (node in pa_q):
                            tokens.extend([var_tok, str(assign[node])])
                            labels.extend([self.ignore_index, self.ignore_index])

                        else:
                            tokens.extend([var_tok, MASK_TOKEN])
                            labels.extend([self.ignore_index, self.ignore_index])
            else:
                # TODO: 確率的な場合はCPTの値と比較する
                # q = self.query_node
                tokens, labels = [], []
                for node in self.topo:
                    var_tok = f"{node}="
                    if node in self.roots:
                        tokens.extend([var_tok, str(assign[node])])
                        labels.extend([self.ignore_index, self.ignore_index])
                    else:
                        if self.chain:
                            pass
                        else:
                            tokens.extend([var_tok, MASK_TOKEN])
                            labels.extend([self.ignore_index, self.tok2id[str(assign[node])]])

            # print(tokens, labels)
            # ['0=', '0', '4=', '<mask>', '2=', '<mask>', '1=', '1', '3=', '0', '5=', '<mask>', '7=', '<mask>', '6=', '<mask>', '8=', '<mask>', '9=', '<mask>']
            # [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4, -100, -100, -100, -100, -100, -100]

            input_ids = [self.tok2id[t] for t in tokens]
            if self.chain:
                labels = input_ids[1:] + [self.tok2id["<eos>"]]
            # label_ids = [self.ignore_index if l == self.ignore_index else self.tok2id[l] for l in labels]
            # pad_len = self.max_len - len(input_ids)

            # input_ids.extend([self.pad_id] * pad_len)
            # labels.extend([self.ignore_index] * pad_len)

            # For debug
            n_roots = len(self.roots)
            input_ids = input_ids[: 2 * (n_roots + 1)]
            labels = labels[: 2 * (n_roots + 1)]


            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class BayesNetTask(GeneralizationTask):
    num_nodes = 20
    config = {
        "name": "bayes_net",
        "description": "Ancestor sampling in Bayesian networks.",
        "data_dir": "data/bayes_net",
        "num_nodes": num_nodes,
        "num_edges": num_nodes * 2,
        "max_length": num_nodes * 2,
        "ignore_index": -100,
    }
    config["vocab_size"] = num_nodes + 5

    def pointwise_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1), ignore_index=self.config["ignore_index"]
        )
        return loss

    def accuracy_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = output.argmax(dim=-1)  # (B, T)
        mask = target != self.config["ignore_index"]

        correct = (pred == target) & mask  # (B, T) bool
        pos_correct = correct.sum(dim=0)  # (T,)
        pos_total = mask.sum(dim=0)  # (T,)

        acc_per_pos = torch.where(
            pos_total > 0,
            pos_correct.float() / pos_total.float(),
            torch.full_like(pos_total, float("nan"), dtype=torch.float),
        )
        return acc_per_pos

        """
        # TODO: 確率的な場合にCoTはサンプリングして評価する必要がある。その平均をaccuracyとする。
        num_valid = mask.sum()
        acc = correct.sum().float() / num_valid.float()
        return acc
        """


if __name__ == "__main__":
    # Example usage
    task = BayesNetTask()
    # dataset = BayesNetOnlineDataset(task.config, split="test")
    dataset = BayesNetOnlineDataset(task.config, deterministic=False, split="train", chain=True)
    # print(f"Number of samples in {dataset.split} set: {len(dataset)}")
    for i, (input_ids, label) in enumerate(dataset):
        if i >= 200:  # Get only the first 5 samples for demonstration
            break
        # print(f"Sample {i+1}: Input IDs: {input_ids}")
        print(input_ids.tolist())
