from .nc1.arithmetic import (
    ArithmeticExpressionDataset,
    ArithmeticExpressionTask,
    ArithmeticExpressionTaskChain,
)
from .nc1.bfvp import BooleanFormulaValueProblemDataset, BooleanFormulaValueProblemTask
from .nc1.word import WordProblemDataset, WordProblemTask, WordProblemTaskChain
from .nc2.path import ReachabilityDataset, ReachabilityTask
from .nc2.strings import (
    EditDistanceTask,
    EditDistanceTaskChain,
    LongestCommonSubsequenceTask,
    PairwiseAlignmentDataset,
)
from .p_complete.cvp import CircuitValueProblemDataset, CircuitValueProblemTask


def get_task_and_datasets(args, chain: bool = False, cot_length: int = None):
    if args.task == "word":
        if chain:
            task = WordProblemTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = WordProblemTask()
        train_dataset = WordProblemDataset(task.config, split="train", chain=chain)
        test_dataset = WordProblemDataset(task.config, split="test", chain=chain)
    elif args.task == "path":
        task = ReachabilityTask(max_input_size=args.input_size)
        train_dataset = ReachabilityDataset(task.config, split="train")
        test_dataset = ReachabilityDataset(task.config, split="test")
    elif args.task == "bfvp":
        task = BooleanFormulaValueProblemTask(max_input_size=args.input_size)
        train_dataset = BooleanFormulaValueProblemDataset(task.config, split="train")
        test_dataset = BooleanFormulaValueProblemDataset(task.config, split="test")
    elif args.task == "arithmetic":
        if chain:
            task = ArithmeticExpressionTaskChain(max_input_size=args.input_size, cot_length=cot_length)
        else:
            task = ArithmeticExpressionTask(max_input_size=args.input_size)
        train_dataset = ArithmeticExpressionDataset(task.config, split="train")
        test_dataset = ArithmeticExpressionDataset(task.config, split="test")
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
    elif args.task == "reg":
        pass
    elif args.task == "fixed_cfg":
        task = CircuitValueProblemTask()
        train_dataset = CircuitValueProblemDataset(task.config, split="train")
        test_dataset = CircuitValueProblemDataset(task.config, split="test")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return task, train_dataset, test_dataset
