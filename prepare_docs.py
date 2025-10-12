import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Dossier des PDFs
DOCS_DIR = "pdfs/"

# Charger tous les PDFs
documents = []
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(DOCS_DIR, filename)
        print(f"Traitement du fichier : {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Erreur avec {pdf_path}: {str(e)}")
            continue

if not documents:
    print("Aucun document valide n'a été chargé.")
    exit()

# Découper en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Embeddings gratuits de Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Créer le vector store FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")  # Sauvegarde locale

print("Vector store créé !")