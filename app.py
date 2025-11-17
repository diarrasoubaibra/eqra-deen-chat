# app.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# === CHARGE TON CODE RAG ===
from main import app as rag_app  # Ton FastAPI est dans main.py

# === HF Spaces utilise app.py comme entrée ===
app = FastAPI()

# Monte ton RAG dans l'app HF
app.mount("/", rag_app)

# === Pour Gradio (optionnel, si tu veux une UI) ===
if os.getenv("USE_GRADIO", "0") == "1":
    import gradio as gr

    def chat_with_bot(message, history):
        from main import rag_chain
        response = rag_chain.invoke(message)
        sources = response.get("sources", [])
        answer = response.get("response", "")
        return f"{answer}\n\n**Sources :**\n" + "\n".join([
            f"- *{s['file']}* (page {s['page']}): `{s['text'][:100]}...`"
            for s in sources
        ])

    demo = gr.ChatInterface(
        fn=chat_with_bot,
        title="Chatbot Islamique RAG",
        description="Pose une question sur le Coran ou les hadiths"
    )
    demo.launch()