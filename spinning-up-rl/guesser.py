import numpy as np
from abc import ABC, abstractmethod


class Rubric(ABC):
    """Base class for reward rubrics."""

    @abstractmethod
    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """Evaluates a batch of guesses and returns a reward vector."""
        pass


class BinaryRubric(Rubric):
    """A rubric that provides reward vectors for entire batches of guesses."""

    def __init__(self, target: int, min: int, max: int):
        self.target = target
        self.min = min
        self.max = max

    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """
        Evaluates a batch of guesses and returns a reward vector.
        The reward vector has the same dimension as the action space.
        """
        action_space_size = self.max - self.min + 1
        reward_vector = [0.0] * action_space_size

        for guess in guesses:
            reward = 1.0 if guess == self.target else 0.0
            action_idx = guess - self.min
            reward_vector[action_idx] += reward

        return [r / len(guesses) for r in reward_vector]


class ExponentialRubric(Rubric):
    """A rubric that provides exponential decay rewards based on distance to target."""

    def __init__(self, target: int, min: int, max: int, decay_factor: float = 5.0):
        self.target = target
        self.min = min
        self.max = max
        self.decay_factor = decay_factor

    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """Evaluates guesses with exponential decay based on distance."""
        action_space_size = self.max - self.min + 1
        reward_vector = [0.0] * action_space_size

        for guess in guesses:
            distance = abs(guess - self.target)
            action_idx = guess - self.min
            reward = np.exp(-distance / self.decay_factor)
            reward_vector[action_idx] += reward

        return [r / len(guesses) for r in reward_vector]


class LinearRubric(Rubric):
    """A rubric that provides linear rewards based on distance to target."""

    def __init__(self, target: int, min: int, max: int):
        self.target = target
        self.min = min
        self.max = max
        self.max_distance = max - min

    def evaluate_batch(self, guesses: list[int]) -> list[float]:
        """Evaluates guesses with linear decay based on distance."""
        action_space_size = self.max - self.min + 1
        reward_vector = [0.0] * action_space_size

        for guess in guesses:
            distance = abs(guess - self.target)
            action_idx = guess - self.min
            reward = 1.0 - (distance / self.max_distance)
            reward_vector[action_idx] += reward

        return [r / len(guesses) for r in reward_vector]


class Guesser:
    def __init__(self, min: int, max: int):
        self.policy = np.ones(max - min + 1) / (max - min + 1)
        self.min = min
        self.max = max

    def guess(self, size: int = 1) -> list[int]:
        """Makes guesses based on the current policy."""
        return np.random.choice(
            list(range(self.min, self.max + 1)), size=size, p=self.policy
        )

    def update_policy(self, reward_vector: list[float]):
        """Updates the policy based on a full reward vector."""
        rewards = np.array(reward_vector)
        self.policy = self.policy + rewards
        self.policy = np.maximum(self.policy, 1e-8)
        self.policy = self.policy / self.policy.sum()


if __name__ == "__main__":
    MIN = 0
    MAX = 10
    TARGET = 6
    BATCH_SIZE = 10
    ITERATIONS = 16

    agent = Guesser(MIN, MAX)
    rubric = BinaryRubric(target=TARGET, min=MIN, max=MAX)

    target_idx = TARGET - MIN
    print(f"Target number: {TARGET}, Probability: {agent.policy[target_idx]:.4f}\n")

    for i in range(ITERATIONS):
        guesses = agent.guess(BATCH_SIZE)
        reward_vector = rubric.evaluate_batch(guesses)
        agent.update_policy(reward_vector)

        if i % 2 == 0:
            print(
                f"Iteration {i + 1}: P(guess={TARGET}) = {agent.policy[target_idx]:.3f}"
            )
