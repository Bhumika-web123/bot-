
import json
from pathlib import Path

FILES = [
    "data/processed/train.jsonl",
    "data/processed/validation.jsonl",
]

def load_records(filename):
    path = Path(filename)

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

all_ok = True

for filename in FILES:
    records = load_records(filename)
    errors = []
    empty_responses = 0
    duplicate_prompts = 0
    seen_prompts = set()

    for index, record in enumerate(records, start=1):
        if not isinstance(record.get("prompt"), list):
            errors.append(f"Example {index}: prompt is not a list")
            continue

        if not isinstance(record.get("completion"), list):
            errors.append(f"Example {index}: completion is not a list")
            continue

        prompt = record["prompt"]
        completion = record["completion"]

        if not prompt or prompt[0].get("role") != "system":
            errors.append(f"Example {index}: missing system message")

        if not prompt or prompt[-1].get("role") != "user":
            errors.append(f"Example {index}: prompt must end with user")

        if len(completion) != 1:
            errors.append(
                f"Example {index}: expected one completion message"
            )
        elif completion[0].get("role") != "assistant":
            errors.append(
                f"Example {index}: completion role must be assistant"
            )
        elif not completion[0].get("content", "").strip():
            empty_responses += 1

        messages = prompt + completion
        for message in messages:
            if not isinstance(message.get("content"), str):
                errors.append(
                    f"Example {index}: message content must be text"
                )
                break

        # Identify duplicate current user prompts.
        user_text = prompt[-1].get("content", "")
        if user_text in seen_prompts:
            duplicate_prompts += 1
        seen_prompts.add(user_text)

    print(f"\n{filename}")
    print("Examples:", len(records))
    print("Distinct current user prompts:", len(seen_prompts))
    print("Repeated current user prompts:", duplicate_prompts)
    print("Empty assistant responses:", empty_responses)
    print("Structural errors:", len(errors))

    for error in errors[:10]:
        print("ERROR:", error)

    if errors or empty_responses:
        all_ok = False

print("\nOverall result:", "PASS" if all_ok else "CHECK ERRORS")
