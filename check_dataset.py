
import json
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def count_tokens(messages, add_generation_prompt=False):
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )

    if hasattr(result, "input_ids"):
        result = result.input_ids

    if hasattr(result, "tolist"):
        result = result.tolist()

    if result and isinstance(result[0], list):
        result = result[0]

    return len(result)

for filename in [
    "data/processed/train.jsonl",
    "data/processed/validation.jsonl",
]:
    records = [
        json.loads(line)
        for line in Path(filename).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    prompt_lengths = []
    full_lengths = []

    for record in records:
        prompt_lengths.append(
            count_tokens(
                record["prompt"],
                add_generation_prompt=True,
            )
        )

        full_lengths.append(
            count_tokens(
                record["prompt"] + record["completion"]
            )
        )

    print(f"\n{filename}")
    print("Examples:", len(records))
    print("Longest prompt:", max(prompt_lengths), "tokens")
    print("Longest full conversation:", max(full_lengths), "tokens")
    print("Average full length:", round(mean(full_lengths), 1))
    print("Examples over limit:",
          sum(n > MAX_LENGTH for n in full_lengths))

    assert all(n > 0 for n in full_lengths), (
        "Invalid token count detected."
    )

print("\nDataset length check complete.")
