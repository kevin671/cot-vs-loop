# タスク名を指名したら対応するものを呼び出す？
# experimetnsいかが適切？


def generate_dataset(task_name: str):
    """
    Generate a dataset for the specified task.

    Args:
        task_name (str): The name of the task for which to generate the dataset.

    Returns:
        Dataset: The generated dataset for the specified task.
    """
    from tasks import task_registry

    if task_name not in task_registry:
        raise ValueError(f"Task '{task_name}' is not registered.")

    task_class = task_registry[task_name]
    return task_class.generate_data()
