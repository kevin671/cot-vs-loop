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

        # self.counts = count_proper_colorings(self.G, self.k)
        self.enumeration = list(enumerate_proper_colorings(self.G, self.k))
        self.counts = len(self.enumeration)

    def _build_vocab(self) -> None:
        """Build a simple token->id mapping for textified graph + color sequences."""
        self.tok2id: Dict[str, int] = {}
        add = lambda tok: self.tok2id.setdefault(tok, len(self.tok2id))

        # special tokens
        add("<pad>")
        add("<sep>")
        add("<state>")
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
                for entry in history:
                    # history entries may be (v, c, a) or (v, c, a, coloring_snapshot)
                    if len(entry) >= 3:
                        v, c, a = entry[0], entry[1], entry[2]
                    else:
                        continue
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
                    # tokens.append("<sep>")

                # Initial coloring
                tokens.append("<sep>")
                for v, col in enumerate(self.input_coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                # mark boundary before history
                input_before_history_len = len(tokens)

                # History: (v,c,a) triples, optionally including a full coloring snapshot
                for entry in history:
                    # tokens.append("<sep>")
                    if len(entry) >= 3:
                        v, c, a = entry[0], entry[1], entry[2]
                    else:
                        continue
                    tokens.append(str(v))
                    tokens.append(f"col{c}")
                    tokens.append("A1" if a else "A0")
                    # If a coloring snapshot is provided, append it after the triplet
                    if len(entry) >= 4:
                        snapshot = entry[3]
                        # tokens.append("<state>")
                        for vv, col in enumerate(snapshot):
                            # tokens.append(f"{vv}=")
                            tokens.append(f"col{col}")

                # Final coloring (target) appended to input so we can train next-token prediction
                # tokens.append("<sep>")
                # for v, col in enumerate(coloring):
                #    tokens.append(f"{v}=")
                #    tokens.append(f"col{col}")

                # Do not include <eos> in the input tokens; keep it only as a label target

                # print(tokens)

                input_ids = torch.tensor([self.tok2id.get(t, self.tok2id["<pad>"]) for t in tokens], dtype=torch.long)

                # labels: next-token prediction, ignore everything before final coloring
                labels = torch.full_like(input_ids, fill_value=self.ignore_index)
                shifted = torch.empty_like(input_ids)
                shifted[:-1] = input_ids[1:]
                shifted[-1] = self.tok2id["<eos>"]
                # we want the model to predict the sequence starting at the first token of final coloring
                # find position of the separator before final coloring: it's the last '<sep>' before final
                # For chain mode, start labels at the beginning of the history
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
                final_start = len(tokens)

                tokens.append("<sep>")
                for v, col in enumerate(coloring):
                    tokens.append(f"{v}=")
                    tokens.append(f"col{col}")

                # print(tokens)

                # Do not include <eos> in the input tokens; keep it only as a label target

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
    # history entries: (v, c, a) or (v, c, a, coloring_snapshot)
    history: List[tuple] = []

    for _ in range(steps):
        res = glauber_step(G, k, coloring, rng)
        # res is (accepted, v, c)
        accepted_flag, v, c = res
        if accepted_flag:
            accepted += 1
        if record_history:
            # store a copy of the current coloring snapshot along with the move
            history.append((v, c, 1 if accepted_flag else 0, coloring.copy()))

    acceptance_rate = accepted / steps
    if record_history:
        return coloring, acceptance_rate, history
    return coloring, acceptance_rate


def count_proper_colorings(G: nx.Graph, k: int) -> int:
    """Count exact number of proper k-colorings via backtracking.

    This is exponential in general and suitable for small graphs (n ~< 20).
    """
    # Implemented via a shared backtracking generator so callers can either
    # enumerate all colorings or just count them.
    return sum(1 for _ in enumerate_proper_colorings(G, k))


def enumerate_proper_colorings(G: nx.Graph, k: int):
    """Generate all proper k-colorings for G.

    Yields lists of length n (where n = number of nodes), giving the color
    assigned to each vertex in the order returned by `list(G.nodes())`.
    This is exponential and intended for small graphs (n ~< 20).
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

    def dfs(pos: int):
        if pos == n:
            yield assigned.copy()
            return
        v = order[pos]
        used = {assigned[u] for u in neighbors[v] if assigned[u] != -1}
        for c in range(k):
            if c in used:
                continue
            assigned[v] = c
            yield from dfs(pos + 1)
            assigned[v] = -1

    yield from dfs(0)


class ColoringTask(GeneralizationTask):
    def __init__(self, n: int, d: int, steps: int, chain: bool = False):
        super().__init__()
        k = 2 * d + 1
        # Estimate vocabulary size and a conservative maximum sequence length.
        # Vocabulary: 6 special tokens + 3 tokens per vertex ("v:", "v=", "v") + k color tokens
        vocab_size = 6 + 3 * n + k

        # Non-chain max length (edges + init + final): n*(d+2) + 1 + 2n + 1 + 2n = n*(d+6) + 2
        max_len_nonchain = n * (d + 6) + 2

        # Chain-mode conservative upper bound:
        # initial part: edges (n*(d+1)) + sep(1) + init(2n) = n*(d+3) + 1
        # per history entry worst-case (with snapshot): 1(<sep>)+1(v)+1(col)+1(Ax)+1(<state>)+2n = 5 + 2n
        # per_entry = 5 + 2 * n
        per_entry = 3 + n
        max_len_chain = n * (d + 3) + 1 + steps * per_entry

        max_length = max(max_len_nonchain, max_len_chain)

        print(f"ColoringTask: n={n}, d={d}, k={k}, vocab_size={vocab_size}, max_length={max_length}")

        self.config = {
            "n": n,
            "d": d,
            "k": k,
            "max_length": int(max_length) if chain else int(max_len_nonchain),
            "vocab_size": int(vocab_size),
            "steps_per_sample": steps,
            "ignore_index": -100,
        }

    # Use default pointwise_loss_fn from GeneralizationTask (cross-entropy per token)


if __name__ == "__main__":
    # Graph parameters
    n = 3  # number of vertices
    d = 2  # degree (low-degree regular)
    seed = 0

    # Color parameter (theoretical safe zone: k >= 2d + 1)
    k = 2 * d + 1

    # MCMC parameters
    steps = 40  # _000

    # Debug
    # Generate d-regular graph
    G = nx.random_regular_graph(d, n, seed=seed)

    # count exact number of proper colorings (for small n)
    # exact_count = count_proper_colorings(G, k)

    enumeration = list(enumerate_proper_colorings(G, k))
    # print(enumeration[0]) # [0, 1, 2]
    exact_count = len(enumeration)

    print(f"Exact number of proper {k}-colorings: {exact_count}")

    print(f"Generated {d}-regular graph with n={n}")
    print(f"Max degree: {max(dict(G.degree()).values())}")
    print(f"Using k={k} colors")

    # Run sequential MCMC chains and collect final colorings
    runs = 50000
    results = []
    for i in range(runs):
        s = seed + i + 1
        final_coloring, acc_rate = run_mcmc(G, k, steps, seed=s)
        results.append((tuple(final_coloring), acc_rate))

    colorings = [r[0] for r in results]

    from collections import Counter

    freq = Counter(colorings)

    print(f"Collected {len(colorings)} samples from MCMC.")

    import matplotlib.pyplot as plt

    keys = list(freq.keys())
    values = [freq[k] for k in keys]
    print("Sampled distribution:", keys, values, flush=True)
    plt.bar(range(len(keys)), values)
    plt.xticks(range(len(keys)), [str(k) for k in keys], rotation=90)
    plt.tight_layout()
    plt.savefig("coloring_histogram.png")

    """
    # Sanity check
    print("Proper coloring:", is_proper(G, final_coloring))

    task = ColoringTask(n, d, steps)

    print(task.config)

    dataset = ColoringsDataset(task.config, split="train", chain=True)  # True)

    # torch.set_printoptions(profile="full")

    for i, (input_ids, labels) in enumerate(dataset):
        print("Input IDs:", input_ids)
        print("Labels:", labels)
        print(input_ids.shape, labels.shape)
        if i >= 0:
            break
    """
