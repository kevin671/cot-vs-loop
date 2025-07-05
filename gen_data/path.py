import argparse
import random
from pathlib import Path

import networkx as nx
import tqdm


def encode_graph_for_transformer(G: nx.Graph, s: int, t: int):
    """
    Encode the graph *G* for a transformer model as a sequence of tokens.
    The sequence is concatenated in the following order:

        [ 'v1', 'v2', ...,          # vertex tokens
          '1,2', '2,3', ...,        # edge tokens (only pairs with u < v)
          's,t' ]                   # query-task token
    """
    vertices = sorted(G.nodes())
    vertex_tokens = [f"v{v}" for v in vertices]
    edge_tokens = [f"{u},{v}" for u, v in sorted(G.edges())]
    task_token = f"{s},{t}"
    reachable = nx.has_path(G, s, t)

    tokens = vertex_tokens + edge_tokens + [task_token]
    return tokens, reachable


def generate_er_reachability_sample(n: int, p: float, seed: int = None):
    """
    Draw a random graph **G(n,p)**, pick a random source/target pair,
    and produce one training example consisting of
    """
    if seed is not None:
        random.seed(seed)

    G = nx.erdos_renyi_graph(n, p, seed=seed)
    s, t = random.sample(list(G.nodes()), 2)
    tokens, reachable = encode_graph_for_transformer(G, s, t)
    label = int(reachable)  # True → 1, False → 0
    return tokens, label


def write_sample(path: str, vertices: list[str], edges: list[str], query: str, label: int):
    """
    Append a single line of tab-separated data to *path*:
    """
    line = " ".join(vertices) + "\t" + " ".join(edges) + "\t" + query + "\t" + str(label) + "\n"
    with open(path, "a") as f:
        f.write(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_nodes", "--n", type=int, default=8, help="number of vertices in G(n,p)")
    p.add_argument("--edge_prob", "--p", type=float, default=None, help="edge probability; defaults to 1.7/num_nodes")
    p.add_argument("--train_size", type=int, default=int(1e6), help="# training samples")
    p.add_argument("--test_size", type=int, default=1000, help="# test samples")
    p.add_argument("--data_dir", type=str, default="data/path", help="directory to drop {train,test}.txt")
    p.add_argument("--seed", type=int, default=42, help="global RNG seed (for reproducibility)")

    args = p.parse_args()

    if args.edge_prob is None:
        args.edge_prob = 1.7 / args.num_nodes

    rng = random.Random(args.seed)
    dpath = Path(args.data_dir, str(args.num_nodes))
    dpath.mkdir(parents=True, exist_ok=True)

    for split, size in [("train", args.train_size), ("test", args.test_size)]:
        out_file = dpath / f"{split}.txt"
        # overwrite if the file already exists
        out_file.write_text("")

        for _ in tqdm.tqdm(range(size), desc=f"Generating {split}"):
            # use an independent seed per sample for determinism yet diversity
            sample_seed = rng.randint(0, 2**32 - 1)

            tokens, label = generate_er_reachability_sample(args.num_nodes, args.edge_prob, seed=sample_seed)

            # split the token sequence back into vertices, edges, and query
            vertices = [tok for tok in tokens if tok.startswith("v")]
            query = tokens[-1]  # last token
            edges = tokens[len(vertices) : -1]  # middle slice

            write_sample(out_file, vertices, edges, query, label)

    print(f"Dataset written to {dpath}")


if __name__ == "__main__":
    main()
