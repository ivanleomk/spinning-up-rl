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
        cleaned = response.replace("标题:", "")[:150]
        responses.add(cleaned)
    if len(responses) >= TOTAL:
        break

print(f"Total unique responses: {len(responses)}")

responses_list = list(responses)

train_full = responses_list[:TRAIN]
test_data = responses_list[TRAIN : TRAIN + EVAL]
rl_data = responses_list[TRAIN + EVAL : TRAIN + EVAL + RL]

SYSTEM_PROMPT = "You are a helpful assistant that reverses Chinese text. Given a Chinese text, output the exact same characters but in reverse order."

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
                "prompt": [SYSTEM_PROMPT] * len(train_subset),
                "completion": train_answers,
            }
        ),
        "test": Dataset.from_dict(
            {
                "question": test_data,
                "answer": test_answers,
                "prompt": [SYSTEM_PROMPT] * len(test_data),
                "completion": test_answers,
            }
        ),
        "rl": Dataset.from_dict(
            {
                "question": rl_data,
                "answer": rl_answers,
                "prompt": [SYSTEM_PROMPT] * len(rl_data),
                "completion": rl_answers,
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
