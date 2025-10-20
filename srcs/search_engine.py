import sqlite3
import chromadb
import config
import openai
import json

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


def expand_query_with_llm(query):
    """Expand the user query using an LLM to include related terms."""
    print(f"\nExpanding query using LLM: '{query}'")
    prompt = f"""
            You are a search query expansion expert.
            Given the user query, output **only** a JSON list (no wrapper object, no extra keys) with 3-5 related search terms.
            Put the original query as the first element.

            User Query: "{query}"

            JSON List Output:
        """
    if not config.USE_API_FOR_LLM or not config.API_KEY:
        raise ValueError("API key is not configured or API usage is disabled.")

    client = openai.OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that strictly outputs JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        reply_content = response.choices[0].message.content
        expanded_queries = json.loads(reply_content)
        # print(f"  - LLM response: {reply_content}")
        if query not in expanded_queries:
            expanded_queries.insert(0, query)
            print(f"  - Expanded queries: {expanded_queries}")
        return expanded_queries
    except Exception as e:
        print(f"  - LLM query expansion failed: {e}")
        return [query]


def hybrid_search(query, top_k=5):
    """Combine keyword and semantic search results."""
    all_queries = expand_query_with_llm(query)

    all_keyword_ids = set()
    all_semantic_ids = set()

    for q in all_queries:
        keyword_results = keyword_search(q, top_k=top_k)
        semantic_results = semantic_search(q, top_k=top_k)
        all_keyword_ids.update(keyword_results)
        all_semantic_ids.update(semantic_results)

    keyword_ids_list = list(all_keyword_ids)
    semantic_ids_list = list(all_semantic_ids)

    fused_scores = {}
    k = 60

    for rank, doc_id in enumerate(keyword_ids_list):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)

    for rank, doc_id in enumerate(semantic_ids_list):
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
