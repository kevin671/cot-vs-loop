from .arithmetic import (
    ArithmeticExpressionDataset,
    ArithmeticExpressionTask,
    ArithmeticExpressionTaskChain,
)
from .dnf import DNFCountDataset, DNFCountTask
from .path import ReachabilityDataset, ReachabilityTask, ReachabilityTaskChain
from .strings import (
    EditDistanceTask,
    EditDistanceTaskChain,
    LongestCommonSubsequenceTask,
    PairwiseAlignmentDataset,
)
from .word import WordProblemDataset, WordProblemTask, WordProblemTaskChain


def get_task_and_datasets(args, chain: bool = False, cot_length: int = None):
    if args.task == "word":
        if chain:
            task = WordProblemTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = WordProblemTask()
        train_dataset = WordProblemDataset(task.config, split="train", chain=chain)
        test_dataset = WordProblemDataset(task.config, split="test", chain=chain)
    elif args.task == "path":
        if chain:
            task = ReachabilityTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = ReachabilityTask(max_input_size=args.input_size)
        train_dataset = ReachabilityDataset(task.config, split="train", chain=chain)
        test_dataset = ReachabilityDataset(task.config, split="test", chain=chain)
    elif args.task == "arithmetic":
        if chain:
            task = ArithmeticExpressionTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = ArithmeticExpressionTask(max_input_size=args.input_size)
        train_dataset = ArithmeticExpressionDataset(task.config, split="train", chain=chain)
        test_dataset = ArithmeticExpressionDataset(task.config, split="test", chain=chain)
    elif args.task == "ed":
        if chain:
            task = EditDistanceTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = EditDistanceTask(max_input_size=args.input_size)
        train_dataset = PairwiseAlignmentDataset(task.config, split="train", chain=chain)
        test_dataset = PairwiseAlignmentDataset(task.config, split="test", chain=chain)
    elif args.task == "lcs":
        task = LongestCommonSubsequenceTask(max_input_size=args.input_size)
        train_dataset = PairwiseAlignmentDataset(task.config, split="train")
        test_dataset = PairwiseAlignmentDataset(task.config, split="test")
    elif args.task == "dnf":
        task = DNFCountTask(input_size=args.input_size)
        train_dataset = DNFCountDataset(task.config, split="train")
        test_dataset = DNFCountDataset(task.config, split="test")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return task, train_dataset, test_dataset
