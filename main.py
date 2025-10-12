from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Clé Hugging Face
huggingface_token  = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configure ton API key OpenRouter
openai_key  = os.getenv("OPENROUTER_API_KEY")

app = FastAPI(title="Chatbot Islamique RAG")

# Charger le vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# LLM gratuit de Hugging Face
# llm = HuggingFacePipeline.from_model_id(
#     model_id="facebook/blenderbot-400M-distill",
#     task="text-generation",
#     # pipeline_kwargs={"max_length": 512, "temperature": 0.7}
# )

# Utiliser OpenRouter via ChatOpenAI
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=openai_key,
    temperature=0.7,
    # max_tokens=512
)

# Prompt template pour le RAG
prompt_template = """
Utilise le contexte suivant pour répondre à la question de manière précise et respectueuse, en te basant sur les connaissances islamiques du Coran et des sources fournies.

Contexte : {context}

Question : {input}

Réponse :
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

# Chaîne pour combiner les documents
combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)

# Chaîne RAG (remplacement de RetrievalQA obsolète)
qa_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    combine_docs_chain=combine_docs_chain
)
# Chaîne RAG
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        result = qa_chain.invoke({"input": query.question})
        return {
            "response": result["answer"],
            "sources": [doc.metadata for doc in result["context"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))