import sys
import streamlit as st

import config
import search_engine

from pathlib import Path
from urllib.parse import quote_plus


# To enable streamlit to locate and import our own module
APP_DIR = Path(__file__).parent
sys.path.append(str(APP_DIR))

# Streamlit Page Configuration
st.set_page_config(
    page_title="Your Personal Research Assistant",
    page_icon="🧪",
    layout="wide",  # use wide mode for more space
)

st.title("Litmus 🧪")
st.markdown(
    """
        Welcome to Litmus, your personal AI-powered research assistant!
        Upload academic papers in PDF format, and let Litmus help you analyze and search through them with ease.
    """
)
st.markdown("---")

query = st.text_input(
    "Enter your search keywords or query:",
    placeholder="e.g., Transformer architecture in NLP",
)
search_button = st.button("Search")

if search_button and query:
    with st.spinner(
        "Searching through your knowledge base... (this may take a moment)"
    ):
        results = search_engine.hybrid_search(query)

    st.subheader(f'Found {len(results)} relevant papers for: "{query}"')

    if not results:
        st.warning(
            "No papers found that meet the relevance threshold. "
            f"You could try a different query or adjust the `SEMANTIC_SEARCH_THRESHOLD` in `config.py` (current value: {config.SEMANTIC_SEARCH_THRESHOLD})."
        )
    else:
        for paper in results:
            with st.expander(
                f"**{paper['title']}** ({paper['conference']} {paper['year']})"
            ):
                st.markdown(f"**Authors:** *{paper['authors']}*")
                st.info(f"**AI Summary:** {paper['generated_summary']}")
                st.markdown(f"**Keywords:** `{paper['keywords']}`")
                # create a google scholar search link for the paper title
                scholar_url = "https://scholar.google.com/scholar?q=" + quote_plus(
                    paper["title"]
                )
                st.markdown(f"[Search for this paper on Google Scholar]({scholar_url})")

                st.markdown(f"--- \n *Local Path: `{paper['file_path']}`*")
elif search_button and not query:
    st.warning("Please enter a search query to proceed.")
