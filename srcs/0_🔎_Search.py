import sys
import streamlit as st
import json

import config
import search_engine

from pathlib import Path
from urllib.parse import quote_plus


# To enable streamlit to locate and import our own module
APP_DIR = Path(__file__).parent
sys.path.append(str(APP_DIR))

# Streamlit Page Configuration
st.set_page_config(page_title="Litmus 🧪 - Search", page_icon="🔎", layout="wide")

st.title("🔎 Search Your Knowledge Base")
st.markdown("Ask a question or enter keywords to find relevant papers and passages.")
st.markdown("---")

query = st.text_input(
    "Enter your search keywords or query:",
    placeholder="e.g., Transformer architecture in NLP",
)
search_button = st.button("Search")

if search_button and query:
    with st.spinner("Performing deep search through your knowledge base..."):
        results = search_engine.hybrid_search(query, top_k=10)

    st.subheader(f'Found {len(results)} relevant papers for: "{query}"')

    if not results:
        st.warning(
            "No relevant passages found that meet the relevance threshold. "
            f"Try a different query or adjust the `SEMANTIC_SEARCH_THRESHOLD` in `config.py` (current: {config.SEMANTIC_SEARCH_THRESHOLD})."
        )
    else:
        for paper in results:
            with st.container(border=True):
                st.subheader(f"📄 {paper['title']}")
                st.caption(
                    f"{paper['conference']} {paper['year']} | Authors: *{paper['authors']}*"
                )
                try:
                    summary_data = json.loads(paper["structured_summary"])
                    st.info(
                        f"**Motivation:** {summary_data.get('motivation', 'N/A')}",
                        icon="🎯",
                    )
                    st.success(
                        f"**Methodology:** {summary_data.get('methodology', 'N/A')}",
                        icon="🛠️",
                    )
                    st.warning(
                        f"**Key Results:** {summary_data.get('key_results', 'N/A')}",
                        icon="💡",
                    )
                except (json.JSONDecodeError, TypeError, AttributeError):
                    st.info("Summary not available in structured format.")

                # st.markdown("**📖 Relevant Passages Found in this Paper:**")
                # for chunk in paper.get("relevant_chunks", []):
                #     if chunk:
                #         st.markdown(f"> ...{chunk.strip()}...")

                st.markdown(f"--- \n *Local Path: `{paper['file_path']}`*")

elif search_button and not query:
    st.error("Please enter a query to search.")
