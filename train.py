"""Character-level encode/decode demo (learning track). Real SFT is finetune.py."""

from tokenizer import encode, decode

text = "morning"

encoded = encode(text)
decoded = decode(encoded)

print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)