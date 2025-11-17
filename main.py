from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import re
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Chatbot Islamique RAG — Avec Versets")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_template("""
Tu es un assistant islamique fiable basé uniquement sur le Coran et la Sunna.
Utilise le contexte fourni, cite les sourates et versets quand nécessaire.

Contexte :
{context}

Question :
{question}

Réponse :
""")

# Formatage des docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Extraction automatique des versets

VERSE_REGEX = r"(\b(?:Sourate\s*)?(\d+)\s*[:]\s*(\d+)\b)"

def extract_verse_refs(text):
    matches = re.findall(VERSE_REGEX, text)
    return [(int(sura), int(aya)) for (_, sura, aya) in matches]

# Récupération du verset dans FAISS
def get_verse_from_faiss(sura, aya):
    query = f"Sourate {sura}:{aya}"
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return None

    best = docs[0]
    return {
        "text": best.page_content,
        "metadata": best.metadata
    }


class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        answer = rag_chain.invoke(query.question)

        verse_refs = extract_verse_refs(answer)

        verses = []
        for sura, aya in verse_refs:
            verse_data = get_verse_from_faiss(sura, aya)
            if verse_data:
                verses.append({
                    "sura": sura,
                    "aya": aya,
                    # "verse": verse_data["text"],
                    # "source": verse_data["metadata"]
                })

        # Rappeler les sources du RAG principal
        sources = retriever.get_relevant_documents(query.question)
        sources_meta = [doc.metadata for doc in sources]

        return {
            "response": answer,
            "verses": verses,   
            "sources": sources_meta 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
