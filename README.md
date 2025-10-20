🚀 Odyssey: A Hybrid AI Travel Assistant for Vietnam

<p align="center">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.12-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython" alt="Python Version"/>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Neo4j-Graph_Database-008CC1%3Fstyle%3Dfor-the-badge%26logo%3Dneo4j" alt="Neo4j"/>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Pinecone-Vector_DB-00BFFF%3Fstyle%3Dfor-the-badge%26logo%3Dpinecone" alt="Pinecone"/>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Groq-LLM_Inference-F05A28%3Fstyle%3Dfor-the-badge" alt="Groq"/>
</p>

This repository contains the source code for Odyssey, a sophisticated, conversational AI travel assistant that provides intelligent travel recommendations and itineraries for Vietnam. Developed as a technical challenge for Blue Enigma Labs, this project transformed a semi-functional script into a robust, production-ready application that showcases a powerful hybrid retrieval architecture.

✨ Key Features

Hybrid RAG System: Combines Pinecone's vector search (for understanding concepts like "romance" or "adventure") with a Neo4j knowledge graph (for factual connections like Located_In or Connected_To).

Advanced AI Reasoning: Implements a Chain-of-Thought (CoT) process where the AI first creates a private, bullet-point plan before writing the final itinerary, leading to more logical and well-structured answers.

Conversational Intelligence:

Intent Detection: Automatically detects whether a user is asking a simple question or requesting a full itinerary.

Conversational Memory: Remembers the context of the conversation, allowing for natural follow-up questions (e.g., answering "yes" to an offer for planning).

Local & Cost-Effective: Uses a local sentence-transformers model for embeddings, making it fast, free to run, and independent of rate-limited APIs.

Robust & Professional: The final code includes comprehensive error handling, a clean project structure, and professional conventions.

🏛️ System Architecture

The application follows a Retrieval-Augmented Generation (RAG) pattern, orchestrating multiple services to produce a high-quality response.

User Query: The user's question is received by the main chat loop.

Local Embedding: The query is converted into a vector embedding locally using a SentenceTransformer model.

Hybrid Retrieval (Parallel):

Vector Search (Pinecone): The query embedding is used to find semantically similar items (cities, attractions) from the Pinecone vector database.

Graph Search (Neo4j): The IDs from the Pinecone results are used to run an intelligent Cypher query in Neo4j, retrieving factual relationships and connections.

Context Augmentation & Prompting: The retrieved data from both databases, along with a summary and a sophisticated system prompt, are compiled into a rich context.

LLM Reasoning (Groq): This context is sent to a high-speed LLM on Groq. The model uses Chain-of-Thought reasoning to first generate a plan and then write the final, coherent answer.

Response to User: The final answer is presented to the user.

🛠️ Tech Stack

AI & Machine Learning:

Language Model: Groq API (llama-3.1-8b-instant)

Embeddings: sentence-transformers (all-MiniLM-L6-v2)

Databases:

Vector Database: Pinecone

Graph Database: Neo4j

Core Libraries:

openai (for Groq API communication)

pinecone-client

neo4j

tqdm (for progress bars)

Environment: Python 3.12, venv for dependency management.

⚙️ Setup and Installation

1. Clone the Repository:

git clone <your-repository-url>
cd <your-repository-name>


2. Create and Activate Virtual Environment:

python -m venv ven
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows


3. Install Dependencies:

pip install -r requirements.txt


4. Configure Your API Keys:

Make a copy of config.py.sample and rename it to config.py.

Open config.py and fill in your secret keys for GROQ_API_KEY, PINECONE_API_KEY, and your Neo4j database credentials.

5. Start Neo4j:

Ensure you have a Neo4j database instance running (e.g., via Neo4j Desktop) and that the credentials match your config.py file.

▶️ How to Run

Execute the scripts in the following order.

Step 1: Load Data into Neo4j
This script populates your graph database with the travel data.

python load_to_neo4j.py


Step 2: Upload Embeddings to Pinecone
This script generates local embeddings and uploads them to your Pinecone index. The first run will download the embedding model.
(Note: Ensure any previous vietnam-travel index in Pinecone is deleted if you are re-running this with new data.)

python pinecone_upload.py


Step 3: (Optional) Visualize the Graph
This script generates an interactive HTML file (neo4j_viz.html) to visualize your graph.

python visualize_graph.py


Step 4: Run the Chat Assistant
Start the interactive chatbot.

export TOKENIZERS_PARALLELISM=false # Recommended to hide tokenizer warnings
python hybrid_chat.py


You can then ask questions like, "create a romantic 4 day itinerary for Vietnam" or "where can I see beaches?".

📸 Demo

Here is a sample interaction with the Odyssey travel assistant.

User:
create a 2 day itinerary for vietnam i want to visit beach also

Odyssey's Response:

=== Odyssey Answer ===

Here is a practical 2-day beach itinerary focusing on the beautiful coastal region of Central Vietnam:

**Day 1: Modern Adventure in Da Nang (city_da_nang)**

* **Morning/Afternoon:** Arrive in Da Nang, a city known for its modern vibe and beautiful beaches. Spend the day relaxing on the sand or exploring some of the local sights. For an adventurous experience, you could visit Da Nang Attraction 181 (attraction_181).
* **Evening:** Enjoy the modern city atmosphere. Your context mentions several hotels like Da Nang Hotel 191 (hotel_191).

**Day 2: Ancient Charm in Hoi An (city_hoi_an)**

* **Morning:** Since the knowledge graph shows that Da Nang 'Is Connected To' Hoi An, take a short trip to the nearby ancient town of Hoi An (city_hoi_an). This city is famous for its romantic atmosphere and heritage.
* **Afternoon:** Explore Hoi An's unique attractions. You could take a romantic boat ride (activity_166) or enjoy the beautiful lanterns the town is known for.
* **Evening:** Depart from Da Nang.

This itinerary is logical and practical as it focuses on two closely connected cities, allowing you to experience both a modern beach city and a historic town within your 2-day timeframe.
