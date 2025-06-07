from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load your training data
dataset = load_dataset("json", data_files="chatbot_training_data.jsonl", split="train")

# Load a small pretrained model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization function
def tokenize(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 👈 this is key!
    return tokenized

# Tokenize your dataset
tokenized = dataset.map(tokenize)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./my_chatbot",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# Start training
trainer.train()

# Save your fine-tuned model
model.save_pretrained("./my_chatbot")
tokenizer.save_pretrained("./my_chatbot")