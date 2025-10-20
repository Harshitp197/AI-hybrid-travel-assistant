# 🚀 Odyssey: A Hybrid AI Travel Assistant for Vietnam

<p align="center">
  <img src="[https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)" alt="Python Version"/>
  <img src="[https://img.shields.io/badge/Neo4j-Graph_Database-008CC1?style=for-the-badge&logo=neo4j](https://img.shields.io/badge/Neo4j-Graph_Database-008CC1?style=for-the-badge&logo=neo4j)" alt="Neo4j"/>
  <img src="[https://img.shields.io/badge/Pinecone-Vector_DB-00BFFF?style=for-the-badge&logo=pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00BFFF?style=for-the-badge&logo=pinecone)" alt="Pinecone"/>
  <img src="[https://img.shields.io/badge/Groq-LLM_Inference-F05A28?style=for-the-badge](https://img.shields.io/badge/Groq-LLM_Inference-F05A28?style=for-the-badge)" alt="Groq"/>
</p>

This repository contains the source code for **Odyssey**, a sophisticated, conversational AI travel assistant that provides intelligent travel recommendations and itineraries for Vietnam. Developed as a technical challenge for Blue Enigma Labs, this project transformed a semi-functional script into a robust, production-ready application that showcases a powerful hybrid retrieval architecture.

## ✨ Key Features

* **Hybrid RAG System:** Combines Pinecone's vector search (for understanding concepts like "romance" or "adventure") with a Neo4j knowledge graph (for factual connections like `Located_In` or `Connected_To`).
* **Advanced AI Reasoning:** Implements a **Chain-of-Thought (CoT)** process where the AI first creates a private, bullet-point plan before writing the final itinerary, leading to more logical and well-structured answers.
* **Conversational Intelligence:**
    * **Intent Detection:** Automatically detects whether a user is asking a simple question or requesting a full itinerary.
    * **Conversational Memory:** Remembers the context of the conversation, allowing for natural follow-up questions (e.g., answering "yes" to an offer for planning).
* **Local & Cost-Effective:** Uses a local `sentence-transformers` model for embeddings, making it fast, free to run, and independent of rate-limited APIs.
* **Robust & Professional:** The final code includes comprehensive error handling, a clean project structure, and professional conventions.

## 🏛️ System Architecture

The application follows a Retrieval-Augmented Generation (RAG) pattern, orchestrating multiple services to produce a high-quality response.

1.  **User Query:** The user's question is received by the main chat loop.
2.  **Local Embedding:** The query is converted into a vector embedding locally using a `SentenceTransformer` model.
3.  **Hybrid Retrieval:**
    * **Vector Search (Pinecone):** The query embedding is used to find semantically similar items from the Pinecone vector database.
    * **Graph Search (Neo4j):** The IDs from the Pinecone results are used to run an intelligent Cypher query in Neo4j, retrieving factual relationships.
4.  **Context Augmentation & Prompting:** The retrieved data, along with a summary and a sophisticated system prompt, are compiled into a rich context.
5.  **LLM Reasoning (Groq):** This context is sent to a high-speed LLM on Groq to generate a plan and then the final, coherent answer.
6.  **Response to User:** The final answer is presented to the user.

## 🛠️ Tech Stack

* **AI & Machine Learning:**
    * Language Model: Groq API (`llama-3.1-8b-instant`)
    * Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Databases:**
    * Vector Database: **Pinecone**
    * Graph Database: **Neo4j**
* **Core Libraries:**
    * `openai` (for Groq API communication)
    * `pinecone-client`
    * `neo4j`
    * `tqdm`
* **Environment:** Python 3.12, `venv`

## ⚙️ Setup and Installation

**1. Clone the Repository:**
```bash
git clone <your-repository-url>
cd <your-repository-name>
