from typing import List, Tuple

from torch.utils.data import Dataset


class DNFCountDataset(Dataset):
    def __init__(self, path: str):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))


class DNFCountTask:
    def __init__(self):
        self.name = "dnf_count"
        self.description = "Given a DNF formula, count the number of satisfying assignments."
        self.input_format = "A DNF formula in the form of a string."
        self.output_format = "An integer representing the number of satisfying assignments."

    def evaluate(self, prediction: int, ground_truth: int) -> bool:
        return prediction == ground_truth
