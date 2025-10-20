import sqlite3
import chromadb
import config
from sentence_transformers import SentenceTransformer


# Globally load the model and database client
print("Search Engine: Loading models and database...")
try:
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name="papers")
    print("Search Engine: Models and vector database loaded successfully.")
except Exception as e:
    print(f"Search Engine: Error loading models or database: {e}")
    embedding_model = None
    collection = None


def get_paper_details_by_ids(ids):
    """Retrieve paper details by IDs from the database."""
    if not ids:
        return []

    placeholders = ", ".join("?" for _ in ids)
    query = f"SELECT * FROM papers WHERE id IN ({placeholders})"

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, ids)
        results = [dict(row) for row in cursor.fetchall()]
        if not results:
            return []
        id_map = {res["id"]: res for res in results}
        return [id_map[id] for id in ids if id in id_map]


def keyword_search(query, top_k=5):
    """Perform keyword search on the vector database."""
    print(f"Keyword Search: Searching for '{query}' with top_k={top_k}...")
    search_term = f"%{query}%"

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
                SELECT id FROM papers
                WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ? OR keywords LIKE ? LIMIT ?
            """,
            (search_term, search_term, search_term, search_term, top_k),
        )

        results = cursor.fetchall()
        return [row["id"] for row in results]


def semantic_search(query, top_k=5):
    """Performing semantic vector search in ChromaDB"""
    if not embedding_model or not collection:
        print(
            "Search Engine: Models or database not loaded. Cannot perform semantic search."
        )
        return []

    print(f"Semantic Search: Searching for '{query}' with top_k={top_k}...")
    query_embedding = embedding_model.encode(query, convert_to_tensor=False).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # filter results by distance threshold
    filtered_ids = []
    ids = results["ids"][0]
    distances = results["distances"][0]

    for doc_id, distance in zip(ids, distances):
        if distance < config.SEMANTIC_SEARCH_THRESHOLD:
            filtered_ids.append(int(doc_id))
        else:
            break

    return filtered_ids


def hybrid_search(query, top_k=5):
    """Combine keyword and semantic search results."""
    keyword_ids = set(keyword_search(query, top_k))
    semantic_ids = set(semantic_search(query, top_k))

    print("\nFusing results with Reciprocal Rank Fusion...")
    fused_scores = {}
    k = 60  # RRF constant

    for rank, doc_id in enumerate(keyword_ids):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)

    for rank, doc_id in enumerate(semantic_ids):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)

    sorted_ids = sorted(
        fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True
    )
    final_ids = sorted_ids[:top_k]
    print(f"Hybrid Search: Found {len(final_ids)} results.")
    return get_paper_details_by_ids(final_ids)


if __name__ == "__main__":
    print("\n--- Testing Search Engine ---")

    test_query = "lora serving"
    results = hybrid_search(test_query, top_k=5)
    print(f"\n--- Top 5 Hybrid Search Results for: '{test_query}' ---")
    if results:
        for i, paper in enumerate(results):
            print(f"\n{i+1}. Title: {paper['title']}")
            print(f"   Conference: {paper['conference']} {paper['year']}")
            print(f"   Summary: {paper['generated_summary']}")
            print(f"   Keywords: {paper['keywords']}")
    else:
        print("No results found.")
