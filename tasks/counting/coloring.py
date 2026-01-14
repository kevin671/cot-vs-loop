import random
from typing import Dict, List

import networkx as nx
import torch
from torch.utils.data import IterableDataset

from tasks.task import GeneralizationTask


class ColoringsDataset(IterableDataset):

    def __init__(self, config: Dict, split: str = "train", chain: bool = False):
        super().__init__()
        self.config = config

        self.n = config["n"]
        self.d = config["d"]

        self.k = 2 * self.d + 1  # number of colors

        self.G = nx.random_regular_graph(self.d, self.n, seed=42)

        self.steps_per_sample = config.get("steps_per_sample", 1000)
        self.chain = chain or bool(config.get("chain", False))

        # tokenization / vocab settings (optional: used to produce text-token inputs)
        self.ignore_index = int(config.get("ignore_index", -100))
        self._build_vocab()

        # Initialization
        self.input_coloring = greedy_init(self.G, self.k)
        self.coloring = self.input_coloring.copy()

        # Precompute edge index tensor for the graph (shape [2, num_edges]).
        edges = list(self.G.edges())
        if edges:
            self.edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            self.edges = torch.empty((2, 0), dtype=torch.long)

    def _build_vocab(self) -> None:
        """Build a simple token->id mapping for textified graph + color sequences."""
        self.tok2id: Dict[str, int] = {}
        add = lambda tok: self.tok2id.setdefault(tok, len(self.tok2id))

        # special tokens
        add("<pad>")
        add("<sep>")
        add("<eos>")
        add("A1")
        add("A0")

        # vertex markers and numeric tokens
        for v in range(self.n):
            add(f"{v}:")
            add(f"{v}=")
            add(str(v))

        # color tokens
        for c in range(self.k):
            add(f"col{c}")

    def __iter__(self):
        # Per-worker seed
        base_seed = torch.initial_seed() % (1 << 30)
        rng = random.Random(base_seed)

        # coloring = self.coloring.copy()

        while True:
            if self.chain:
                coloring, _, history = run_mcmc(
                    self.G,
                    self.k,
                    steps=self.steps_per_sample,
                    seed=rng.randint(0, 1 << 30),
                    init=self.coloring,  # in-place modification
                    record_history=True,
                )
            else:
                coloring, _ = run_mcmc(
                    self.G,
                    self.k,
                    steps=self.steps_per_sample,
                    seed=rng.randint(0, 1 << 30),
                    init=self.coloring,
                )

            # self.coloring = coloring

            if self.chain:
                # Input: (initial_coloring, edge_index, history_seq)
                # Build a single sequence of triplets: (v,c,a),(v,c,a),...
                hist_list: list[int] = []
                for v, c, a in history:
                    hist_list.extend([v, c, a])

                history_seq = torch.tensor(hist_list, dtype=torch.long)

                # Build token sequence similar to DNF dataset: edges, init, history, final
                tokens = []

                # Edges: for each vertex, add an index marker and its neighbors, separated
                for v in range(self.n):
                    tokens.append(f"{v}:")
                    nbrs = sorted(list(self.G.neighbors(v)))
                    for u in nbrs:
                        tokens.append(str(u))
                    tokens.append("<sep>")

                # Initial coloring
                tokens.append("<sep>")
                for v, col in enumerate(self.input_coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                # mark boundary before history
                input_before_history_len = len(tokens)

                # History: (v,c,a) triples
                tokens.append("<sep>")
                for v, c, a in history:
                    tokens.append(str(v))
                    tokens.append(f"col{c}")
                    tokens.append("A1" if a else "A0")

                # Final coloring (target) appended to input so we can train next-token prediction
                tokens.append("<sep>")
                for v, col in enumerate(coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                tokens.append("<eos>")

                input_ids = torch.tensor([self.tok2id.get(t, self.tok2id["<pad>"]) for t in tokens], dtype=torch.long)

                # labels: next-token prediction, ignore everything before final coloring
                labels = torch.full_like(input_ids, fill_value=self.ignore_index)
                shifted = torch.empty_like(input_ids)
                shifted[:-1] = input_ids[1:]
                shifted[-1] = self.tok2id["<eos>"]
                # we want the model to predict the sequence starting at the first token of final coloring
                # find position of the separator before final coloring: it's the last '<sep>' before final
                final_start = None
                for i in range(len(tokens) - 1, -1, -1):
                    if tokens[i] == "<sep>":
                        final_start = i + 1
                        break
                if final_start is None:
                    final_start = input_before_history_len

                labels[final_start:] = shifted[final_start:]

                yield input_ids, labels
            else:
                # Input side: (initial_coloring, edge_index), Target: final coloring
                # Non-chain: edges + init + final coloring appended
                tokens = []
                for v in range(self.n):
                    tokens.append(f"{v}:")
                    nbrs = sorted(list(self.G.neighbors(v)))
                    for u in nbrs:
                        tokens.append(str(u))
                    tokens.append("<sep>")

                tokens.append("<sep>")
                for v, col in enumerate(self.input_coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                # mark start of final coloring
                final_start = len(tokens) + 1
                tokens.append("<sep>")
                for v, col in enumerate(coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                tokens.append("<eos>")

                input_ids = torch.tensor([self.tok2id.get(t, self.tok2id["<pad>"]) for t in tokens], dtype=torch.long)
                labels = torch.full_like(input_ids, fill_value=self.ignore_index)
                shifted = torch.empty_like(input_ids)
                shifted[:-1] = input_ids[1:]
                shifted[-1] = self.tok2id["<eos>"]
                labels[final_start:] = shifted[final_start:]

                yield input_ids, labels


# Utility functions


def is_proper(G: nx.Graph, coloring: List[int]) -> bool:
    """Check whether a coloring is proper."""
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            return False
    return True


def greedy_init(G: nx.Graph, k: int) -> List[int]:
    """
    Simple greedy algorithm to obtain an initial proper k-coloring.
    Requires k >= Delta + 1.
    """
    n = G.number_of_nodes()
    coloring = [-1] * n

    for v in range(n):
        used = {coloring[u] for u in G.neighbors(v) if coloring[u] != -1}
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            raise ValueError("Greedy initialization failed. Increase k.")

    return coloring


# Glauber dynamics (MCMC)


def glauber_step(G: nx.Graph, k: int, coloring: List[int], rng: random.Random):
    """
    One step of Glauber dynamics for proper k-colorings.

    Returns True iff the move is accepted (state changes).
    """
    n = G.number_of_nodes()

    # 1. choose a vertex and a color uniformly at random
    v = rng.randrange(n)
    c = rng.randrange(k)

    # optional: avoid trivial self-loop
    if c == coloring[v]:
        return False, v, c

    # 2. check if recoloring is proper
    for u in G.neighbors(v):
        if coloring[u] == c:
            return False, v, c

    coloring[v] = c
    return True, v, c


def run_mcmc(
    G: nx.Graph,
    k: int,
    steps: int,
    seed: int = 0,
    init: List[int] | None = None,
    record_history: bool = False,
):
    """
    Run M(G, k) for a given number of steps.
    """
    rng = random.Random(seed)

    if init is None:
        coloring = greedy_init(G, k)
    else:
        coloring = init.copy()
        if not is_proper(G, coloring):
            raise ValueError("Initial coloring is not proper.")

    accepted = 0
    history: List[tuple] = []

    for _ in range(steps):
        res = glauber_step(G, k, coloring, rng)
        # res is (accepted, v, c)
        accepted_flag, v, c = res
        if accepted_flag:
            accepted += 1
        if record_history:
            history.append((v, c, 1 if accepted_flag else 0))

    acceptance_rate = accepted / steps
    if record_history:
        return coloring, acceptance_rate, history
    return coloring, acceptance_rate


def count_proper_colorings(G: nx.Graph, k: int) -> int:
    """Count exact number of proper k-colorings via backtracking.

    This is exponential in general and suitable for small graphs (n ~< 20).
    """
    # Map nodes to indices 0..n-1
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # Build neighbor index sets
    neighbors = [set(idx[u] for u in G.neighbors(v)) for v in nodes]

    # Order vertices by degree (largest first) for better pruning
    order = sorted(range(n), key=lambda i: len(neighbors[i]), reverse=True)

    assigned = [-1] * n

    def dfs(pos: int) -> int:
        if pos == n:
            return 1
        v = order[pos]
        used = {assigned[u] for u in neighbors[v] if assigned[u] != -1}
        total = 0
        for c in range(k):
            if c in used:
                continue
            assigned[v] = c
            total += dfs(pos + 1)
            assigned[v] = -1
        return total

    return dfs(0)


class ColoringTask(GeneralizationTask):
    def __init__(self, n: int, d: int):
        super().__init__()
        k = 2 * d + 1
        self.config = {
            "n": n,
            "d": d,
            "k": k,
            "max_length": n,
            "vocab_size": k,
            "ignore_index": -100,
        }

    # Use default pointwise_loss_fn from GeneralizationTask (cross-entropy per token)


if __name__ == "__main__":
    # Graph parameters
    n = 4  # number of vertices
    d = 2  # degree (low-degree regular)
    seed = 0

    # Color parameter (theoretical safe zone: k >= 2d + 1)
    k = 2 * d + 1

    # MCMC parameters
    steps = 1000  # _000

    # Debug
    # Generate d-regular graph
    G = nx.random_regular_graph(d, n, seed=seed)

    # count exact number of proper colorings (for small n)
    exact_count = count_proper_colorings(G, k)
    print(f"Exact number of proper {k}-colorings: {exact_count}")

    print(f"Generated {d}-regular graph with n={n}")
    print(f"Max degree: {max(dict(G.degree()).values())}")
    print(f"Using k={k} colors")

    # Run Glauber dynamics
    final_coloring, acc_rate = run_mcmc(G, k, steps, seed=seed)

    print("Final coloring (first 20 vertices):", final_coloring[:20])
    print("Acceptance rate:", acc_rate)

    # Sanity check
    print("Proper coloring:", is_proper(G, final_coloring))
