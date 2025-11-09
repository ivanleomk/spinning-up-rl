from guesser import Guesser, BinaryRubric, Rubric, ExponentialRubric, LinearRubric
from typing import Type, TypedDict
import matplotlib.pyplot as plt


class ExperimentResult(TypedDict):
    label: str
    probs: list[float]


def run_experiment(
    min_val: int,
    max_val: int,
    target: int,
    batch_size: int,
    iterations: int,
    rubric_class: Type[Rubric] = BinaryRubric,
) -> list[float]:
    """Run a single experiment and return probability trajectory."""
    agent = Guesser(min_val, max_val)
    rubric = rubric_class(target=target, min=min_val, max=max_val)
    target_idx = target - min_val

    probs = [agent.policy[target_idx]]

    for _ in range(iterations):
        guesses = agent.guess(batch_size)
        reward_vector = rubric.evaluate_batch(guesses)
        agent.update_policy(reward_vector)
        probs.append(agent.policy[target_idx])

    return probs


def run_action_space_ablation(
    target: int,
    batch_size: int,
    iterations: int,
    action_spaces: list[int],
    rubric_class: Type[Rubric] = BinaryRubric,
) -> list[ExperimentResult]:
    results = []
    for action_space_size in action_spaces:
        min_val = 0
        max_val = action_space_size - 1
        probs = run_experiment(
            min_val, max_val, target, batch_size, iterations, rubric_class
        )
        results.append({"label": f"Action Space: {action_space_size}", "probs": probs})
    return results


def run_batch_size_ablation(
    min_val: int,
    max_val: int,
    target: int,
    iterations: int,
    batch_sizes: list[int],
    rubric_class: Type[Rubric] = BinaryRubric,
) -> list[ExperimentResult]:
    results = []
    for batch_size in batch_sizes:
        probs = run_experiment(
            min_val, max_val, target, batch_size, iterations, rubric_class
        )
        results.append({"label": f"Batch Size: {batch_size}", "probs": probs})
    return results


def run_reward_function_ablation(
    min_val: int,
    max_val: int,
    target: int,
    batch_size: int,
    iterations: int,
    rubrics: list[tuple[Type[Rubric], str]],
) -> list[ExperimentResult]:
    results = []
    for rubric_class, label in rubrics:
        probs = run_experiment(
            min_val, max_val, target, batch_size, iterations, rubric_class
        )
        results.append({"label": label, "probs": probs})
    return results


def plot_ablation(results: list[ExperimentResult], title: str, filepath: str):
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result["probs"], label=result["label"])
    plt.xlabel("Iteration")
    plt.ylabel("Probability of Target")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    TARGET = 5
    BATCH_SIZE = 50
    ITERATIONS = 50
    ACTION_SPACES = [9, 99, 999]
    results = run_action_space_ablation(TARGET, BATCH_SIZE, ITERATIONS, ACTION_SPACES)
    plot_ablation(results, "Effect of Action Space Size", "action_space_rl.png")

    BATCH_SIZES = [50, 100, 200]
    results = run_batch_size_ablation(0, 1000, TARGET, ITERATIONS, BATCH_SIZES)
    plot_ablation(results, "Effect of Batch Size", "batch_size_rl.png")

    rubrics = [
        (BinaryRubric, "Binary Reward"),
        (ExponentialRubric, "Exponential Reward"),
        (LinearRubric, "Linear Reward"),
    ]
    results = run_reward_function_ablation(
        0, 100, target=50, batch_size=50, iterations=100, rubrics=rubrics
    )
    plot_ablation(results, "Effect of Reward Function", "reward_function_rl.png")
