
import json
import random
from pathlib import Path

DATA_PATH = Path("data/Conversation_1.json")
OUTPUT_DIR = Path("data/processed")

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and concise assistant."
)

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42


def load_conversations(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Expected a list of conversation groups.")

    for group in data:
        if "conversation" not in group:
            raise ValueError("Missing 'conversation' field.")

        if not isinstance(group["conversation"], list):
            raise ValueError("'conversation' must be a list.")

    return data


def convert_group(group):
    """Create one training record per user/bot turn."""
    history = []
    records = []

    for pair in group["conversation"]:
        user_text = str(pair.get("user", "")).strip()
        bot_text = str(pair.get("bot", "")).strip()

        if not user_text or not bot_text:
            print("Skipping an empty user/bot pair.")
            continue

        # Include earlier turns and the current user message.
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]

        completion = [
            {"role": "assistant", "content": bot_text}
        ]

        records.append({
            "prompt": prompt,
            "completion": completion,
        })

        # Preserve the context for the next turn.
        history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ])

    return records


def main():
    groups = load_conversations(DATA_PATH)

    # Split by conversation group, not individual turns.
    shuffled = list(groups)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    validation_count = max(
        1, round(len(shuffled) * VALIDATION_FRACTION)
    )

    validation_groups = shuffled[:validation_count]
    training_groups = shuffled[validation_count:]

    train_records = [
        record
        for group in training_groups
        for record in convert_group(group)
    ]

    validation_records = [
        record
        for group in validation_groups
        for record in convert_group(group)
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_jsonl(records, filename):
        path = OUTPUT_DIR / filename

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(f"Saved {len(records)} records to {path}")

    save_jsonl(train_records, "train.jsonl")
    save_jsonl(validation_records, "validation.jsonl")

    print("\nDataset preparation complete.")
    print(f"Training groups: {len(training_groups)}")
    print(f"Validation groups: {len(validation_groups)}")
    print(f"Training examples: {len(train_records)}")
    print(f"Validation examples: {len(validation_records)}")


if __name__ == "__main__":
    main()
