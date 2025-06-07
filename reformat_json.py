import json

# === Step 1: Load your comment JSON ===
with open("comments.json") as f:
    raw_data = json.load(f)

# === Step 2: Convert each comment into a prompt-response pair ===
data = []
for item in raw_data:
    comments_per_pr = list(item.values())[0]
    for comment in comments_per_pr:
        entry = {
            "text": f"User: What would this engineer say in a code review?\nBot: {comment}"
        }
        data.append(entry)

# === Step 3: Save as a .jsonl file (one JSON object per line) ===
with open("chatbot_training_data.jsonl", "w") as out_file:
    for entry in data:
        json.dump(entry, out_file)
        out_file.write("\n")

print(f"✅ Created chatbot_training_data.jsonl with {len(data)} examples.")