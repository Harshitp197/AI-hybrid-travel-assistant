# hybrid_chat.py
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import config
import functools
import traceback # For better error logging

# -----------------------------
# Configuration
# -----------------------------
CHAT_MODEL = "llama-3.1-8b-instant" # Verified model from Groq docs
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize Clients & Models
# -----------------------------
try:
    groq_client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    embed_model = SentenceTransformer('all-MiniLM-L6-v2') # Local embedding model
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    # Test Neo4j connection
    driver.verify_connectivity()
    print("Successfully connected to Neo4j.")
except Exception as e:
    print(f"ERROR: Failed to initialize clients or connect to databases: {e}")
    print(traceback.format_exc())
    exit(1) # Exit if essential clients fail to initialize

# -----------------------------
# Helper Functions
# -----------------------------

@functools.lru_cache(maxsize=128)
def embed_text(text: str) -> List[float]:
    """Generates sentence embeddings using a local model, with caching."""
    try:
        return embed_model.encode(text).tolist()
    except Exception as e:
        print(f"ERROR generating embedding: {e}")
        return [] # Return empty list on error

def pinecone_query(query_text: str, top_k: int = TOP_K) -> Optional[List[Dict[str, Any]]]:
    """Queries Pinecone index using a local embedding model. Returns None on error."""
    try:
        query_embedding = embed_text(query_text)
        if not query_embedding: # Handle embedding error
            return None
            
        res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return res.get("matches", [])
    except Exception as e:
        print(f"ERROR during Pinecone query: {e}")
        return None

def summarize_results(matches: List[Dict[str, Any]]) -> str:
    """Uses the LLM to create a concise summary of top Pinecone search results."""
    if not matches:
        return "No relevant items found in the vector database."

    summary_context = "### Top Search Results:\n"
    for i, match in enumerate(matches[:3]): # Summarize top 3
        meta = match.get('metadata', {})
        description = meta.get('text', 'No description.')
        tags = ", ".join(meta.get('tags', []))
        summary_context += (
            f"{i+1}. {meta.get('name', 'N/A')} (id: {match.get('id', 'N/A')})\n"
            f"   - Type: {meta.get('type', 'N/A')}\n"
            f"   - Desc: {description[:100]}...\n" # Shorter truncation
            f"   - Tags: {tags}\n"
        )

    summary_prompt = [
        {"role": "system", "content": "Summarize the key themes or locations from the provided search results in one concise sentence."},
        {"role": "user", "content": f"{summary_context}\n\nSummarize the main points:"}
    ]

    try:
        summary_completion = groq_client.chat.completions.create(
            messages=summary_prompt, model=CHAT_MODEL, temperature=0.3, max_tokens=100
        )
        return summary_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"ERROR generating search summary: {e}")
        return "Could not generate summary for search results."

def fetch_graph_context(node_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetches relevant context from Neo4j based on node IDs from Pinecone."""
    if not node_ids:
        return []
        
    cities = [nid for nid in node_ids if nid.startswith('city_')]
    other_entities = [nid for nid in node_ids if not nid.startswith('city_')]
    
    facts = []
    cypher_query = """
    UNWIND $cities AS city_id
    MATCH (city:City {id: city_id})<-[r:Located_In]-(place)
    RETURN city.name AS source, 'Contains' AS rel, place.id AS target_id,
           place.name AS target_name, place.description AS target_desc
    LIMIT 10 // Limit results per city

    UNION

    UNWIND $cities AS city_id
    MATCH (city1:City {id: city_id})-[r:Connected_To]-(city2:City)
    WHERE city2.id IN $cities // Only show connections between relevant cities
    RETURN city1.name AS source, 'Is Connected To' AS rel, city2.id AS target_id,
           city2.name AS target_name, city2.description AS target_desc

    UNION

    UNWIND $other_entities AS entity_id
    MATCH (entity {id: entity_id})-[r]-(neighbor)
    RETURN entity.id AS source, type(r) AS rel, neighbor.id AS target_id,
           neighbor.name AS target_name, neighbor.description AS target_desc
    LIMIT 10 // Limit results for other entities
    """
    try:
        with driver.session() as session:
            result = session.run(cypher_query, cities=cities, other_entities=other_entities)
            # Use a set to avoid duplicate facts if relationships are bidirectional
            unique_facts = set()
            for record in result:
                # Create a frozenset of items to check uniqueness regardless of order for connections
                fact_tuple = tuple(sorted(record.items()))
                if fact_tuple not in unique_facts:
                    facts.append(dict(record))
                    unique_facts.add(fact_tuple)
        return facts
    except Exception as e:
        print(f"ERROR fetching graph context: {e}")
        return []

def build_prompt(user_query: str, pinecone_matches: List[Dict[str, Any]], graph_facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Builds the prompt for the LLM, including context and instructions."""
    system_prompt = (
        "You are 'Odyssey', an expert travel concierge specializing ONLY in Vietnam travel planning and information based on the provided context. " # Added specialization
        "Your primary goal is to accurately answer the user's travel questions or create itineraries using ONLY the provided context. "
        "Follow these rules strictly:\n"
        "1. **Scope Rule:** If the user's query is NOT related to Vietnam travel (e.g., asking about cooking, politics, science, other countries), politely state that you are a Vietnam travel assistant and cannot answer questions outside that topic. Do NOT attempt to answer out-of-scope questions.\n" # <<< NEW RULE
        "2. Base your travel-related answers ONLY on the information provided in the 'CONTEXT' section.\n"
        "3. Do NOT use any external knowledge about Vietnam or real-world travel times.\n"
        "4. If you mention a specific Vietnam location, attraction, or activity from the context, cite its ID in parentheses, e.g., 'Hoi An (city_hoi_an)'.\n"
        "5. If the user's request sounds like they want a plan or itinerary for Vietnam, create one using the context.\n"
        "6. **Crucial Travel Rule:** Ensure itineraries are realistic. Only suggest travel between cities if the Knowledge Graph context shows a 'Is Connected To' relationship. Avoid long-distance travel on short itineraries.\n"
        "7. Use provided descriptions and tags to justify choices and make them appealing.\n"
        "8. If the user asks a simple travel question, provide a concise answer first. After answering, ask 'Would you like help planning an itinerary related to this?'"
    )

    search_summary = summarize_results(pinecone_matches)

    vec_context = "### Vector Search Results (Semantic Ideas):\n"
    if not pinecone_matches:
        vec_context += "No relevant items found.\n"
    else:
        for match in pinecone_matches:
            meta = match.get('metadata', {})
            description = meta.get('text', 'No description available.')
            tags = ", ".join(meta.get('tags', []))
            vec_context += (
                f"- {meta.get('name', 'N/A')} (id: {match.get('id', 'N/A')})\n"
                f"  - Type: {meta.get('type', 'N/A')}\n"
                f"  - Description: {description}\n"
                f"  - Tags: {tags}\n"
            )

    graph_context = "\n### Knowledge Graph Connections (Factual Relationships):\n"
    if not graph_facts:
        graph_context += "No direct factual connections found for the retrieved items.\n"
    else:
        for fact in graph_facts:
            source = fact.get('source', 'Unknown Node')
            rel = fact.get('rel', 'related to')
            target_name = fact.get('target_name', 'Unknown Node')
            target_id = fact.get('target_id', 'N/A')
            graph_context += f"- '{source}' {rel} '{target_name}' (id: {target_id}).\n"

    user_prompt = (
        f"CONTEXT:\n### Search Summary:\n{search_summary}\n\n"
        f"{vec_context}\n{graph_context}\n\n"
        f"TASK:\nUsing only the context above and following all rules, respond to this user's request: '{user_query}'"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def call_chat_with_reasoning(prompt_messages: List[Dict[str, str]], user_query: str) -> str:
    """Uses a two-step Chain-of-Thought process (Plan -> Write) for itinerary generation."""
    try:
        # === STEP 1: The Planner ===
        planning_prompt = prompt_messages + [
            {"role": "user", "content": (
                "Based on the user's request and the context, create a bullet-point plan for the itinerary. "
                "Justify choices using descriptions/tags and respect the travel rules."
            )}
        ]
        plan_completion = groq_client.chat.completions.create(
            messages=planning_prompt, model=CHAT_MODEL, temperature=0.5
        )
        plan = plan_completion.choices[0].message.content

        # === STEP 2: The Writer ===
        writing_prompt = prompt_messages + [
            {"role": "assistant", "content": plan}, # Feed the plan back to the AI
            {"role": "user", "content": "Using your plan above, write the final, polished travel itinerary."}
        ]
        final_completion = groq_client.chat.completions.create(
            messages=writing_prompt, model=CHAT_MODEL, temperature=0.7, max_tokens=1024
        )
        return final_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"ERROR during Chain-of-Thought call: {e}")
        return "Sorry, I encountered an error while planning the itinerary. Please try again."

def call_direct_answer(prompt_messages: List[Dict[str, str]]) -> str:
    """Calls the LLM for a direct answer and asks the follow-up planning question."""
    try:
        # Append the specific instruction for direct answer + follow-up
        direct_answer_prompt = prompt_messages + [
            {"role": "user", "content": "Answer the user's question directly and concisely based ONLY on the context. Then, ask 'Would you like help planning an itinerary related to this?'"}
        ]
        answer_completion = groq_client.chat.completions.create(
            messages=direct_answer_prompt, model=CHAT_MODEL, temperature=0.3, max_tokens=300 # Shorter response needed
        )
        return answer_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"ERROR during direct answer call: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again."

# -----------------------------
# Interactive Chat Loop
# -----------------------------
def interactive_chat():
    """Handles the main chat loop, including user input, state management, and calling LLM."""
    print("\nHybrid travel assistant 'Odyssey' is ready. Type 'exit' to quit.")

    # --- Memory Variables ---
    last_query_for_planning: Optional[str] = None
    last_matches_for_planning: Optional[List[Dict[str, Any]]] = None
    last_graph_facts_for_planning: Optional[List[Dict[str, Any]]] = None
    asked_about_planning: bool = False
    # -----------------------

    while True:
        try:
            query: str = input("\nEnter your travel question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "bye"):
                print("Goodbye! Safe travels.")
                break

            # --- Check for Follow-up FIRST ---
            if asked_about_planning and query.lower() in ("yes", "y", "sure", "ok", "please", "okay", "yes please"):
                if last_query_for_planning and last_matches_for_planning is not None and last_graph_facts_for_planning is not None:
                    print("Okay, planning an itinerary based on your previous question...")
                    # Rebuild the prompt using the *stored* original query and context
                    prompt = build_prompt(last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning)
                    # Call the planning function directly
                    answer = call_chat_with_reasoning(prompt, last_query_for_planning)

                    # Reset memory *after* successfully planning
                    last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning = None, None, None
                    asked_about_planning = False
                else:
                    # Memory was lost somehow
                    answer = "I seem to have lost the context. Could you please ask your original question again?"
                    last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning = None, None, None
                    asked_about_planning = False

            # --- If not a follow-up, process as a New Query ---
            else:
                # Reset memory/flags for any query that isn't a direct 'yes' confirmation
                asked_about_planning = False
                last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning = None, None, None

                print("Searching knowledge base...")
                matches = pinecone_query(query)
                if matches is None: # Handle Pinecone connection error
                    answer = "Sorry, I encountered an issue connecting to the knowledge base. Please try again."
                else:
                    match_ids = [m["id"] for m in matches] if matches else []
                    graph_facts = fetch_graph_context(match_ids)
                    prompt = build_prompt(query, matches, graph_facts)

                    # Simple Intent Check
                    is_planning_request = any(keyword in query.lower() for keyword in ["itinerary", "plan", "days", "trip", "visit", "go", "travel"])

                    if is_planning_request:
                        print("Understood, planning an itinerary...")
                        answer = call_chat_with_reasoning(prompt, query)
                        # Ensure memory is clear after a direct planning request
                        asked_about_planning = False
                        last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning = None, None, None
                    else:
                        print("Finding a direct answer...")
                        answer = call_direct_answer(prompt)

                        # Check if the answer contains the specific follow-up phrase
                        if "Would you like help planning an itinerary related to this?" in answer:
                             # Store context ONLY if the bot asked the follow-up
                             last_query_for_planning = query
                             last_matches_for_planning = matches
                             last_graph_facts_for_planning = graph_facts
                             asked_about_planning = True
                             print("DEBUG: Stored context for potential planning follow-up.") # Optional debug print
                        else:
                             # If the bot didn't ask, ensure memory is clear
                             asked_about_planning = False
                             last_query_for_planning, last_matches_for_planning, last_graph_facts_for_planning = None, None, None


            # --- Print Final Answer ---
            print("\n=== Odyssey Answer ===\n")
            print(answer)
            print("\n====================\n")

        except KeyboardInterrupt:
             print("\nGoodbye! Safe travels.")
             break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print(traceback.format_exc())
            print("Please try again.")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    try:
        interactive_chat()
    finally:
        # Ensure Neo4j driver is closed properly on exit
        if 'driver' in globals() and driver:
            driver.close()
            print("\nNeo4j connection closed.")