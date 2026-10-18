# this is my parser that extracts the bot an user pair from my json data
import json
with open(
    r"C:\Users\Bhumika\Desktop\chatbot\data\Conversation_1.json","r",encoding="utf-8"
) as f:
    data=json.load(f)

pairs = []

for example in data:
    for conversation in example["conversation"]:
        pairs.append(conversation)

training_data = []

for pair in pairs:
    user_text = pair["user"]
    bot_text = pair["bot"]

    training_text = "User: " + user_text + "\nBot: " + bot_text

    training_data.append(training_text)

print("Number of training samples:", len(training_data))
print("\nFirst sample:")
print(training_data[0])



# cheching if i have any blank dialouges 
empty_samples = []

for i, sample in enumerate(training_data):
    if not sample.strip():
        empty_samples.append(i)

print("Empty samples:", len(empty_samples))



#find out length of our data here ;)
sample_lengths = []

for sample in training_data:
    sample_lengths.append(len(sample))

print("Shortest sample:", min(sample_lengths))
print("Longest sample:", max(sample_lengths))
print("Average length:", sum(sample_lengths) / len(sample_lengths))

#checking for duplicates
unique_samples = set(training_data)

print("Total samples:", len(training_data))
print("Unique samples:", len(unique_samples))
print("Duplicate samples:", len(training_data) - len(unique_samples))

# cecking if it contains both side of conversation

missing_user = []
missing_bot = []

for i, sample in enumerate(training_data):
    if not sample.startswith("User:"):
        missing_user.append(i)

    if "\nBot:" not in sample:
        missing_bot.append(i)

print("Missing User:", len(missing_user))
print("Missing Bot:", len(missing_bot))

for i, sample in enumerate(training_data):
    if "\nBot:" not in sample:
        missing_bot.append(i)

print("Missing User:", len(missing_user))
print("Missing Bot:", len(missing_bot))

# Creating input and target
inputs = []
targets = []

for pair in pairs:
    inputs.append("User: " + pair["user"])
    targets.append("Bot: " + pair["bot"])

print("Input:", inputs[0])
print("Target:", targets[0])

# checking if everthing is converted
print("Number of inputs:", len(inputs))
print("Number of targets:", len(targets))

print("\nFirst 3 inputs:")
for i in range(3):
    print(inputs[i])

print("\nFirst 3 targets:")
for i in range(3):
    print(targets[i])
    
# THIS IS WHERE DA TOKENIZER STARTS (I would like to pat myself for not giving up)
# Combine all training text
all_text = "\n".join(training_data)

# Find every unique character
vocab = sorted(set(all_text))

# Vocabulary size
vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)
print("Vocabulary:", vocab)

stoi={ch:i for i,ch in enumerate(vocab)}
itos={i:ch for i,ch in enumerate(vocab)}
print("stoi:",stoi)
print("itos:",itos)

def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)

with open("data/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
    
with open("data/vocab.json", "r", encoding="utf-8") as f:
    saved_vocab = json.load(f)

print(saved_vocab)
print("Vocabulary size:", len(saved_vocab))