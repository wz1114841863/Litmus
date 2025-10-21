import sys
import streamlit as st
import sqlite3
import json
import pandas as pd

import config

from pathlib import Path
from urllib.parse import quote_plus

# To enable streamlit to locate and import our own module
APP_DIR = Path(__file__).parent.parent
sys.path.append(str(APP_DIR))

st.set_page_config(
    page_title="Browse Papers - Litmus",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Browse Your Paper Collection")
st.markdown(
    """
    Select a conference and year to see all the papers you've analyzed.
    """
)
st.markdown("---")


@st.cache_data
def load_all_analyzed_papers():
    """Load all analyzed papers from the database."""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            query = "SELECT * FROM papers"
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return pd.DataFrame()


df_papers = load_all_analyzed_papers()

if df_papers.empty:
    st.warning(
        "No analyzed papers found in the database. Please run the AI Analysis Engine first."
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        conferences = df_papers["conference"].unique().tolist()
        selected_conference = st.selectbox("Select Conference", options=conferences)

    df_conference = df_papers[df_papers["conference"] == selected_conference]

    with col2:
        years = sorted(df_conference["year"].unique(), reverse=True)
        selected_year = st.selectbox("Select a Year:", options=years)

    df_final_selection = df_conference[df_conference["year"] == selected_year]

    st.markdown("---")
    st.header(
        f"Displaying {len(df_final_selection)} papers from {selected_conference} {selected_year}"
    )

st.header(
    f"Displaying {len(df_final_selection)} papers from {selected_conference} {selected_year}"
)

for _, paper in df_final_selection.iterrows():
    with st.expander(f"**{paper['title']}**"):
        st.markdown(f"**Authors:** *{paper['authors']}*")

        try:
            if paper.get("structured_summary") and pd.notna(
                paper["structured_summary"]
            ):
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
            else:
                st.info("Structured summary not yet generated for this paper.")
        except (json.JSONDecodeError, TypeError):
            st.info(
                f"Could not parse summary: {paper.get('structured_summary', 'N/A')}"
            )

        st.markdown("**Keywords:**")
        try:
            if paper.get("keywords") and pd.notna(paper["keywords"]):
                keywords_data = json.loads(paper["keywords"])
                if keywords_data.get("author"):
                    st.write(
                        "Authored:",
                        " | ".join(f"`{kw}`" for kw in keywords_data["author"]),
                    )
                if keywords_data.get("generative"):
                    st.write(
                        "AI Concepts:",
                        " | ".join(f"`{kw}`" for kw in keywords_data["generative"]),
                    )
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
