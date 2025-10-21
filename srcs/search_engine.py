import sqlite3
import chromadb
import config
import openai
import json

import config

from collections import defaultdict
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
                WHERE title LIKE ?  OR authors LIKE ? OR keywords LIKE ? LIMIT ?
            """,
            (search_term, search_term, search_term, top_k),
        )

        results = cursor.fetchall()
        return [row["id"] for row in results]


def semantic_search(query, top_k=5):
    """Performing semantic vector search in chunks ChromaDB"""
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

    relevant_chunks = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i in range(len(ids)):
        if (distances[i] < config.SEMANTIC_SEARCH_THRESHOLD) and documents[i]:
            relevant_chunks.append(
                {
                    "paper_id": metadatas[i]["paper_id"],
                    "chunk_text": documents[i],
                    "distance": distances[i],
                }
            )
    return relevant_chunks


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
    all_keyword_paper_ids = set()
    all_relevant_chunks = []

    for q in all_queries:
        all_keyword_paper_ids.update(keyword_search(q, top_k=top_k))
        all_relevant_chunks.extend(semantic_search(q, top_k=top_k))

    paper_scores = defaultdict(float)
    paper_chunks = defaultdict(list)

    for chunk in all_relevant_chunks:
        paper_id = chunk["paper_id"]
        score = 1 / (chunk["distance"] + 0.1)
        paper_scores[paper_id] += score
        if chunk["chunk_text"] not in paper_chunks[paper_id]:
            paper_chunks[paper_id].append(chunk["chunk_text"])

    for paper_id in all_keyword_paper_ids:
        paper_scores[paper_id] += config.KEYWORD_SEARCH_BOOST

    if not paper_scores:
        return []

    sorted_paper_ids = sorted(
        paper_scores.keys(), key=lambda pid: paper_scores[pid], reverse=True
    )
    top_paper_ids = sorted_paper_ids[:top_k]
    final_results = get_paper_details_by_ids(top_paper_ids)

    for paper in final_results:
        paper["relevant_chunks"] = paper_chunks.get(paper["id"], [])

    print(f"\nReturning top {len(final_results)} fused and ranked results.")
    return final_results


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
