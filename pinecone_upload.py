# pinecone_upload.py
import json
import time
import traceback
from typing import List, Dict, Any, Generator, Tuple

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
import config

# --- Constants ---
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Configuration Check ---
required_configs = [
    'PINECONE_API_KEY', 'PINECONE_INDEX_NAME', 'PINECONE_VECTOR_DIM',
    'PINECONE_CLOUD', 'PINECONE_REGION'
]
missing_configs = [conf for conf in required_configs if not hasattr(config, conf)]
if missing_configs:
    raise ValueError(f"Missing required configuration variables in config.py: {', '.join(missing_configs)}")

INDEX_NAME: str = config.PINECONE_INDEX_NAME
VECTOR_DIM: int = config.PINECONE_VECTOR_DIM

# -----------------------------
# Initialize Clients & Model
# -----------------------------
try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    print("Pinecone client initialized.")

except Exception as e:
    print(f"ERROR: Failed to initialize embedding model or Pinecone client: {e}")
    print(traceback.format_exc())
    exit(1)

# -----------------------------
# Helper Functions
# -----------------------------

def get_or_create_pinecone_index(index_name: str, dimension: int, metric: str = "cosine") -> Index:
    """Gets the Pinecone index object, creating the index if it doesn't exist."""
    try:
        # *** THE FIX IS HERE: Added () to .names() ***
        existing_indexes: List[str] = pc.list_indexes().names()
        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating new index...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION
                )
            )
            print("Waiting for index to initialize...")
            time.sleep(10)
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")

        index = pc.Index(index_name)
        print(f"Connected to Pinecone index: '{index_name}'.")
        return index

    except Exception as e:
        print(f"ERROR interacting with Pinecone index: {e}")
        print(traceback.format_exc())
        raise

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts using the local SentenceTransformer model."""
    try:
        return embed_model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        print(f"ERROR generating embeddings: {e}")
        return [[] for _ in texts]

def chunked_data(iterable: List[Any], n: int) -> Generator[List[Any], None, None]:
    """Yields successive n-sized chunks from a list."""
    if not isinstance(iterable, list):
        iterable = list(iterable)
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def load_and_prepare_data(filepath: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Loads data from JSON file and prepares it for embedding and upserting."""
    items_to_upsert: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nodes: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(nodes)} nodes from {filepath}.")

        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                print(f"WARN: Skipping node due to missing 'id': {node.get('name', 'N/A')}")
                continue

            semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
            if not semantic_text.strip():
                print(f"WARN: Skipping node '{node_id}' due to empty text for embedding.")
                continue

            metadata: Dict[str, Any] = {
                "id": node_id,
                "type": node.get("type"),
                "name": node.get("name"),
                "city": node.get("city", node.get("region", "")),
                "tags": node.get("tags", []),
                "text": semantic_text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            items_to_upsert.append((node_id, semantic_text, metadata))

        print(f"Prepared {len(items_to_upsert)} items for upserting.")
        return items_to_upsert

    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}")
        raise
    except Exception as e:
        print(f"ERROR loading or preparing data: {e}")
        print(traceback.format_exc())
        raise

# -----------------------------
# Main Upload Logic
# -----------------------------
def main():
    """Main function to load data, generate embeddings, and upsert to Pinecone."""
    start_time = time.time()
    try:
        pinecone_index = get_or_create_pinecone_index(INDEX_NAME, VECTOR_DIM)
        items = load_and_prepare_data(DATA_FILE)

        if not items:
            print("No valid items to upsert. Exiting.")
            return

        print(f"Starting upsert process in batches of {BATCH_SIZE}...")

        for batch in tqdm(list(chunked_data(items, BATCH_SIZE)), desc="Uploading batches"):
            ids: List[str] = [item[0] for item in batch]
            texts: List[str] = [item[1] for item in batch]
            metadatas: List[Dict[str, Any]] = [item[2] for item in batch]

            embeddings: List[List[float]] = get_embeddings(texts)

            valid_vectors = []
            for i, emb in enumerate(embeddings):
                if emb:
                    valid_vectors.append({"id": ids[i], "values": emb, "metadata": metadatas[i]})
                else:
                    print(f"WARN: Skipping vector for id '{ids[i]}' due to embedding error.")
            
            if not valid_vectors:
                print(f"WARN: No valid vectors in batch starting with id '{ids[0]}'. Skipping.")
                continue

            try:
                pinecone_index.upsert(vectors=valid_vectors)
            except Exception as upsert_error:
                print(f"\nERROR during Pinecone upsert for batch starting with id '{ids[0]}': {upsert_error}")

            time.sleep(0.1)

        print("\nAll items processed.")

    except Exception as e:
        print(f"\nAn error occurred during the main upload process: {e}")
        print(traceback.format_exc())
    finally:
        end_time = time.time()
        print(f"Script finished in {end_time - start_time:.2f} seconds.")

# -----------------------------
# Script Execution Guard
# -----------------------------
if __name__ == "__main__":
    main()
