import math
import random
from typing import List, Tuple

import numpy as np


def generate_random_nand_circuit(G: int, r: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Generate a random NAND circuit with G gates and r input nodes.
    Returns:
        - gate_inputs: List of (in1, in2) gate input indices (could be input or other gates).
        - input_assignments: values for inputs y1 to yr (0 or 1)
    """
    gate_inputs = []
    input_assignments = [random.randint(0, 1) for _ in range(r)]

    for g in range(G):
        choices = list(range(r)) + list(range(G))  # Inputs or gates
        in1 = random.choice(choices)
        in2 = random.choice(choices)
        gate_inputs.append((in1, in2))

    return gate_inputs, input_assignments


def eval_nand_circuit(gate_inputs: List[Tuple[int, int]], input_vals: List[int]) -> List[int]:
    """
    Evaluate the NAND circuit and return gate outputs.
    """
    r = len(input_vals)
    outputs = []

    for g, (in1, in2) in enumerate(gate_inputs):
        x = input_vals[in1] if in1 < r else outputs[in1 - r]
        y = input_vals[in2] if in2 < r else outputs[in2 - r]
        outputs.append(1 - (x & y))  # NAND
    return outputs


def build_iterated_mod_instance(
    gate_inputs: List[Tuple[int, int]], input_vals: List[int]
) -> Tuple[int, List[int], int]:
    """
    Construct (a, [b1,...,b2G], label) from NAND circuit.
    """
    G = len(gate_inputs)
    r = len(input_vals)

    # 1. Assign bit vector a (length 2G+1)
    a_bits = [1] * (2 * G + 1)
    for g, (in1, in2) in enumerate(gate_inputs):
        edge1 = 2 * (g + 1)
        edge2 = 2 * (g + 1) - 1
        for edge, inp in zip([edge1, edge2], [in1, in2]):
            if inp < r:
                a_bits[edge] = input_vals[inp]

    a = sum(b << i for i, b in enumerate(a_bits))

    # 2. Build Og and bi values
    out_edges = [[] for _ in range(G)]
    for g, (in1, in2) in enumerate(gate_inputs):
        for inp_idx in [in1, in2]:
            if inp_idx >= r:
                out_edges[inp_idx - r].append(2 * (g + 1))
                out_edges[inp_idx - r].append(2 * (g + 1) - 1)

    b_list = []
    for g in range(G):
        edge1 = 2 * (g + 1)
        edge2 = 2 * (g + 1) - 1
        b1 = 2**edge2
        b2 = 2**edge1 + 2**edge2 + sum(2**j for j in out_edges[g])
        b_list.extend([b1, b2])

    # 3. Compute iterated mod
    x = a
    for b in b_list:
        x = x % b

    label = 1 if x == 0 else 0
    return a, b_list, label


def generate_dataset(num_samples: int, G: int = 5, r: int = 3) -> List[str]:
    """
    Generate dataset in the form: "a b1 b2 ... b2G <sep> label"
    """
    dataset = []
    for _ in range(num_samples):
        gates, inputs = generate_random_nand_circuit(G, r)
        a, b_list, label = build_iterated_mod_instance(gates, inputs)
        sample = f"{a} " + " ".join(map(str, b_list)) + f" <sep> {label}"
        dataset.append(sample)
    return dataset


def save_dataset(path: str, dataset: List[str]):
    with open(path, "w") as f:
        for line in dataset:
            f.write(line + "\n")
