import json
from pathlib import Path
from transformers import AutoTokenizer

from constants import MODEL_ID, RAW_CONVERSATIONS, TRAIN_JSONL

# Load Qwen's actual tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if RAW_CONVERSATIONS.exists():
    with RAW_CONVERSATIONS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    first_pair = data[0]["conversation"][0]
    user_text = first_pair["user"]
    bot_text = first_pair["bot"]
else:
    record = json.loads(Path(TRAIN_JSONL).read_text(encoding="utf-8").splitlines()[0])
    user_text = record["prompt"][-1]["content"]
    bot_text = record["completion"][0]["content"]

print("USER:")
print(user_text)

print("\nBOT:")
print(bot_text)

# Create a chat conversation
messages = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": bot_text}
]

# Apply Qwen's chat format
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print("\nFORMATTED FOR QWEN:")
print(formatted_text)

# Tokenize it
tokens = tokenizer(
    formatted_text,
    return_tensors="pt"
)

print("\nTOKEN IDs:")
print(tokens["input_ids"])

print("\nNUMBER OF TOKENS:")
print(tokens["input_ids"].shape[1])

# Decode back to text
decoded = tokenizer.decode(
    tokens["input_ids"][0],
    skip_special_tokens=False
)

print("\nDECODED:")
print(decoded)