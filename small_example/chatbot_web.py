import gradio as gr
import os

def chat_fn(message, history=[]):
    reply = "This is a test response."  # Dummy response
    history.append((message, reply))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Message")
    clear = gr.Button("Clear")
    state = gr.State([])

    msg.submit(chat_fn, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

# Launch with external access
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))