FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install gradio==3.50.2

CMD ["python", "chatbot_web.py"]