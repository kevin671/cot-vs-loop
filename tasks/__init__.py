from .nc1.arithmetic import ArithmeticExpressionDataset, ArithmeticExpressionTask
from .nc1.bfvp import BooleanFormulaValueProblemDataset, BooleanFormulaValueProblemTask
from .nc1.word import WordProblemDataset, WordProblemTask
from .nc2.path import ReachabilityDataset, ReachabilityTask
from .nc2.strings import (
    EditDistanceTask,
    EditDistanceTaskChain,
    LongestCommonSubsequenceTask,
    PairwiseAlignmentDataset,
)


def get_task_and_datasets(args, chain: bool = False):
    if args.task == "word":
        task = WordProblemTask()
        train_dataset = WordProblemDataset(task.config, split="train")
        test_dataset = WordProblemDataset(task.config, split="test")
    elif args.task == "path":
        task = ReachabilityTask(max_input_size=args.input_size)
        train_dataset = ReachabilityDataset(task.config, split="train")
        test_dataset = ReachabilityDataset(task.config, split="test")
    elif args.task == "bfvp":
        task = BooleanFormulaValueProblemTask(max_input_size=args.input_size)
        train_dataset = BooleanFormulaValueProblemDataset(task.config, split="train")  # TODO:
        test_dataset = BooleanFormulaValueProblemDataset(task.config, split="test")
    elif args.task == "arithmetic":
        task = ArithmeticExpressionTask(max_input_size=args.input_size)
        train_dataset = ArithmeticExpressionDataset(task.config, split="train")
        test_dataset = ArithmeticExpressionDataset(task.config, split="test")
    elif args.task == "ed":
        if chain:
            task = EditDistanceTaskChain(max_input_size=args.input_size)
        else:
            task = EditDistanceTask(max_input_size=args.input_size)
        train_dataset = PairwiseAlignmentDataset(task.config, split="train", chain=chain)
        test_dataset = PairwiseAlignmentDataset(task.config, split="test", chain=chain)
    elif args.task == "lcs":
        task = LongestCommonSubsequenceTask(max_input_size=args.input_size)
        train_dataset = PairwiseAlignmentDataset(task.config, split="train")
        test_dataset = PairwiseAlignmentDataset(task.config, split="test")
    elif args.task == "reg":
        pass
    elif args.task == "fixed_cfg":
        pass
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return task, train_dataset, test_dataset
