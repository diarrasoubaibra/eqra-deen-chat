# Islamic RAG API — FastAPI + LangChain + FAISS + OpenRouter

Une API intelligente permettant de poser des questions sur l’Islam à partir de documents authentiques (PDF indexés).  
Le système combine :

- **LangChain v0.3+ (API récente et stable)**
- **FAISS** pour la recherche vectorielle
- **Sentence-Transformers** (embeddings)
- **OpenRouter (Mistral / Llama / autres modèles)**
- **FastAPI**
- **HuggingFace Spaces** pour l’hébergement
sdk: docker
app_port: 8000
pinned: false
license: mit
---

## Fonctionnalités

### RAG performant  
Basé sur FAISS + embeddings modernes (all-MiniLM), rapide et léger.

### Réponses justifiées et sourcées  
Chaque réponse renvoie :

- Le texte généré
- Les PDF utilisés
- Les pages exactes référencées
- **Les versets cités automatiquement extraits du Coran**

### Compatible API externe  
D’autres développeurs peuvent consommer ton API sans accéder au code source.

### Déployable : Render / HuggingFace / Railway  
HuggingFace recommandé (rapide + gratuit + simple).

---

## Endpoints

### `POST /chat`

#### Request
```json
{
  "question": "Quels sont les droits de la femme selon le Coran ?"
}

Réponse
{
  "answer": "… réponse générée …",
  "sources": [
    {
      "source": "pdfs/le_coran_et_la_science.pdf",
      "page": 12,
      "page_label": "13"
    }
  ],
  "verses": [
    {
      "reference": "Sourate 2:228",
      "text": "… extrait du verset …"
    }
  ]
}

Pipeline — Comment ça marche ?

La question arrive sur /chat

Le retrieveur FAISS sélectionne les passages PDF les plus pertinents

Le LLM génère une réponse augmentée par les documents

Le backend détecte automatiquement les références « Sourate X:Y »

Il va extraire ces versets depuis les documents PDF indexés

Le tout est renvoyé sous forme d’un JSON propre

#Structure du projet
/
│── main.py
│── requirements.txt
│── faiss_index/        # Base vectorielle
│── pdfs/               # Facultatif (pour la préparation)
│── README.md

##Installation locale
1️⃣ Cloner le projet
git clone https://github.com/TON-NOM/islamic-rag-api.git
cd islamic-rag-api

## 2️⃣ Créer un environnement virtuel
python -m venv env
source env/bin/activate     # Linux/macOS
env\Scripts\activate        # Windows

3️⃣ Installer les dépendances
pip install -r requirements.txt

4️⃣ Ajouter les variables d’environnement

Créer un fichier .env :
OPENROUTER_API_KEY=ta_cle_openrouter

5️⃣ Lancer l’API
uvicorn main:app --reload
➡️ L’API est prête sur :
http://localhost:8000