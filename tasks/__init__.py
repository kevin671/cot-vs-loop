from .nc1.bfvp import BooleanFormulaValueProblemDataset, BooleanFormulaValueProblemTask
from .nc1.word import WordProblemDataset, WordProblemTask
from .nc2.path import ReachabilityDataset, ReachabilityTask
from .nc2.strings import (
    EditDistanceTask,
    LongestCommonSubsequenceTask,
    PairwiseAlignmentDataset,
)


def get_task_and_datasets(args):
    if args.task == "word":
        task = WordProblemTask()
        train_dataset = WordProblemDataset(task.config, split="train")
        test_dataset = WordProblemDataset(task.config, split="test")
    elif args.task == "path":
        task = ReachabilityTask()
        train_dataset = ReachabilityDataset(task.config, split="train")
        test_dataset = ReachabilityDataset(task.config, split="test")
    elif args.task == "bfvp":
        task = BooleanFormulaValueProblemTask()
        train_dataset = BooleanFormulaValueProblemDataset(task.config, split="train")
        test_dataset = BooleanFormulaValueProblemDataset(task.config, split="test")
    elif args.task == "ed":
        task = EditDistanceTask()
        train_dataset = PairwiseAlignmentDataset(task.config, split="train")
        test_dataset = PairwiseAlignmentDataset(task.config, split="test")
    elif args.task == "lcs":
        task = LongestCommonSubsequenceTask()
        train_dataset = PairwiseAlignmentDataset(task.config, split="train")
        test_dataset = PairwiseAlignmentDataset(task.config, split="test")
    elif args.task == "reg":
        pass
    elif args.task == "fixed_cfg":
        pass
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return task, train_dataset, test_dataset
