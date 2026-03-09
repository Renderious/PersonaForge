import chromadb
from chromadb.config import Settings
import uuid

# Initialize ChromaDB client.
# Using ephemeral storage for simplicity and lightweight execution,
# but it could be easily switched to persistent storage.
client = chromadb.Client(Settings(is_persistent=False))

def get_or_create_collection(collection_name: str):
    """
    Retrieves an existing ChromaDB collection or creates a new one.

    Args:
        collection_name (str): The name of the collection (e.g., the book title).

    Returns:
        chromadb.Collection: The collection instance.
    """
    # Chroma collection names must be a certain format.
    # Replace spaces or invalid chars with underscores.
    safe_name = "".join([c if c.isalnum() else "_" for c in collection_name]).strip("_")

    try:
        return client.get_collection(name=safe_name)
    except Exception:
        return client.create_collection(name=safe_name)


def store_chunks(collection_name: str, chunks: list[str], metadata: list[dict] = None):
    """
    Stores a list of text chunks into the specified ChromaDB collection.

    Args:
        collection_name (str): The name of the collection to store the chunks in.
        chunks (list[str]): The text chunks to store.
        metadata (list[dict], optional): A list of metadata dicts to attach to each chunk.
    """
    collection = get_or_create_collection(collection_name)

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Add chunks to the collection in batches if necessary, but for a simple
    # implementation, we can add them directly. Chroma handles batching internally up to a point.
    if metadata:
        collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
    else:
        collection.add(
            documents=chunks,
            ids=ids
        )


def query_chunks(collection_name: str, query: str, n_results: int = 3, character_filter: str = None) -> list[str]:
    """
    Queries the ChromaDB collection for chunks relevant to the given query.

    Args:
        collection_name (str): The name of the collection to query.
        query (str): The search query (e.g., the user's chat message).
        n_results (int): The number of top chunks to return.
        character_filter (str): The character name to filter the search to.

    Returns:
        list[str]: A list of the most relevant text chunks.
    """
    collection = get_or_create_collection(collection_name)

    # If the collection is empty, querying will fail or return empty.
    if collection.count() == 0:
        return []

    query_kwargs = {
        "query_texts": [query],
        "n_results": min(n_results, collection.count())
    }

    if character_filter:
        query_kwargs["where"] = {"characters": {"$contains": character_filter}}

    results = collection.query(**query_kwargs)

    # The documents are returned in a nested list structure: [['chunk1', 'chunk2']]
    if results and results.get('documents') and len(results['documents']) > 0:
        return results['documents'][0]

    return []
