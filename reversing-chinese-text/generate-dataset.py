import random

from datasets import load_dataset, Dataset, DatasetDict
from tenacity import retry, stop_after_attempt, wait_exponential

TRAIN = 10000
TRAIN_SIZES = [100, 500, 2500, 5000, 10000]
EVAL = 2500
RL = 2500
TOTAL = TRAIN + EVAL + RL

dataset = load_dataset("Iess/chinese_modern_poetry", streaming=True)

responses = set()
for row in dataset["train"]:
    response = row["response"]
    if response:
        cleaned = response.replace("标题:", "")
        target_len = random.randint(250, 300)
        if len(cleaned) < target_len:
            repeats = (target_len // len(cleaned)) + 1
            cleaned = (cleaned * repeats)[:target_len]
        else:
            cleaned = cleaned[:target_len]
        responses.add(cleaned)
    if len(responses) >= TOTAL:
        break

print(f"Total unique responses: {len(responses)}")

responses_list = list(responses)

train_full = responses_list[:TRAIN]
test_data = responses_list[TRAIN : TRAIN + EVAL]
rl_data = responses_list[TRAIN + EVAL : TRAIN + EVAL + RL]

SYSTEM_MESSAGE = (
    "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
)


def build_prompt(question: str) -> list[dict]:
    return [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": question, "role": "user"},
    ]


def build_completion(answer: str) -> list[dict]:
    return [
        {
            "content": f"<reversed_text>\n{answer}\n</reversed_text>",
            "role": "assistant",
        }
    ]


for size in TRAIN_SIZES:
    train_subset = train_full[:size]
    train_answers = [t[::-1] for t in train_subset]
    test_answers = [t[::-1] for t in test_data]
    rl_answers = [t[::-1] for t in rl_data]

    splits = {
        "train": Dataset.from_dict(
            {
                "question": train_subset,
                "answer": train_answers,
                "prompt": [build_prompt(q) for q in train_subset],
                "completion": [build_completion(a) for a in train_answers],
            }
        ),
        "test": Dataset.from_dict(
            {
                "question": test_data,
                "answer": test_answers,
                "prompt": [build_prompt(q) for q in test_data],
                "completion": [build_completion(a) for a in test_answers],
            }
        ),
        "rl": Dataset.from_dict(
            {
                "question": rl_data,
                "answer": rl_answers,
                "prompt": [build_prompt(q) for q in rl_data],
                "completion": [build_completion(a) for a in rl_answers],
            }
        ),
    }
    dataset_dict = DatasetDict(splits)
    print(f"Pushing dataset with train size {size}")
    print(dataset_dict)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def push_with_retry():
        dataset_dict.push_to_hub(f"ivanleomk/reverse-chinese-poetry-{size}")

    push_with_retry()
