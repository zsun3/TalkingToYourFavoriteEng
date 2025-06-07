import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Load your fine-tuned model and tokenizer
model_path = "./my_chatbot"  # <- your trained model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chatbot function
def chat(user_input, history=[]):
    # Format conversation history if you want context
    prompt = f"User: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = response.split("Bot:")[-1].strip()
    history.append((user_input, bot_reply))
    return history, history

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("Talking to your favorite Engineer...")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question like you're talking to your favorite Engineer")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

# Launch it
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))