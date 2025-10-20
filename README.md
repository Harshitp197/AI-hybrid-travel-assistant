Hybrid AI Travel Assistant for Vietnam

This project is a sophisticated, conversational AI travel assistant that provides intelligent travel recommendations and itineraries for Vietnam. It was developed as a technical challenge for Blue Enigma Labs, transforming a semi-functional script into a robust, production-ready application.

The assistant leverages a powerful hybrid retrieval architecture, combining the strengths of semantic vector search with a structured knowledge graph to deliver context-aware, logical, and high-quality responses.

https://www.google.com/search?q=https://github.com/harshit-budhraja/AI-Hybrid-Chat-for-Blue-Enigma/assets/103609825/2569502b-a384-46c5-a6e5-4d0812386926

Key Features

Hybrid Retrieval System: Combines Pinecone's vector search (for understanding concepts like "romance" or "adventure") with a Neo4j knowledge graph (for factual connections like Located_In or Connected_To).

Advanced AI Reasoning: Implements a Chain-of-Thought (CoT) process where the AI first creates a bullet-point plan before writing the final itinerary, leading to more logical and well-structured answers.

Conversational Intelligence:

Intent Detection: Automatically detects whether a user is asking a simple question or requesting a full itinerary.

Conversational Memory: Remembers the context of the conversation, allowing for natural follow-up questions (e.g., answering "yes" to an offer for planning).

Local & Cost-Effective: Uses a local sentence-transformers model for embeddings, making it fast, free to run, and independent of rate-limited APIs.

Robust & Professional: The final code includes comprehensive error handling, a clean project structure, and professional conventions.

Tech Stack

AI & Machine Learning:

Language Model: Groq API (interfaced via openai library)

Embeddings: sentence-transformers (local model: all-MiniLM-L6-v2)

Databases:

Vector Database: Pinecone

Graph Database: Neo4j

Core Libraries:

openai (for Groq API communication)

pinecone-client

neo4j

tqdm (for progress bars)

Environment: Python 3.12, venv for dependency management.

Setup and Installation

1. Clone the Repository:

git clone <your-repository-url>
cd <your-repository-name>


2. Create and Activate Virtual Environment:

python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows


3. Install Dependencies:

pip install -r requirements.txt


4. Configure Your API Keys:

Make a copy of config.py.sample and rename it to config.py.

Open config.py and fill in your secret keys for:

GROQ_API_KEY

PINECONE_API_KEY

Your Neo4j database credentials (NEO_URI, NEO_USER, NEO_PASSWORD).

5. Start Neo4j:

Ensure you have a Neo4j database instance running (e.g., via Neo4j Desktop) and that the credentials match your config.py file.

How to Run

Execute the scripts in the following order:

Step 1: Load Data into Neo4j
This script populates your graph database with the travel data.

python load_to_neo4j.py


Step 2: Upload Embeddings to Pinecone
This script generates local embeddings and uploads them to your Pinecone index. The first run will download the embedding model. Note: Ensure any previous vietnam-travel index in Pinecone is deleted if you are re-running this with new data.

python pinecone_upload.py


Step 3: (Optional) Visualize the Graph
This script generates an interactive HTML file (neo4j_viz.html) to visualize your graph.

python visualize_graph.py


Step 4: Run the Chat Assistant
Start the interactive chatbot.

python hybrid_chat.py


You can then ask questions like, "create a romantic 4 day itinerary for Vietnam" or "where can I see beaches?".