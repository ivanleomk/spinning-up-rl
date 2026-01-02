import verifiers as vf
from datasets import load_dataset

TRAIN_SIZES = [100, 500, 2500, 5000, 10000]


def load_environment(
    train_size: int = 100,
    system_prompt: str
    | None = "Reverse the text character-by-character. Put your answer in <reversed_text> tags.",
    **kwargs,
) -> vf.Environment:
    """
    Loads a custom environment.
    """

    if train_size not in TRAIN_SIZES:
        raise ValueError(f"Invalid train size: {train_size}. Choose from {TRAIN_SIZES}")

    train_dataset = load_dataset(
        f"ivanleomk/reverse-chinese-poetry-{train_size}", split="rl"
    )
    eval_dataset = load_dataset(
        f"ivanleomk/reverse-chinese-poetry-{train_size}", split="test"
    )
    parser = vf.XMLParser(["reversed_text"], answer_field="reversed_text")

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        """

        def lcs_ratio(x: str, y: str) -> float:
            """
            Return the longest common subsequence ratio of x and y.
            """
            from difflib import SequenceMatcher

            return SequenceMatcher(None, x, y).ratio()

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
        ],
        weights=[1.0],
        parser=parser,
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
