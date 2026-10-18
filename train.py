from tokenizer import encode, decode

text = "morning"

encoded = encode(text)
decoded = decode(encoded)

print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)