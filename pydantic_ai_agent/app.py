import streamlit as st
from agent_utils import get_search_results

# ------------------------- Page Config ------------------------- #
st.set_page_config(
    page_title="🔎 GenAI Search Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------- Custom CSS ------------------------- #
st.markdown("""
    <style>
        .main {
            background-color: #0f1117;
            color: white;
        }
        .stTextInput>div>div>input {
            color: white;
            background-color: #1e1f26;
        }
        .stButton>button {
            background-color: #c0392b;
            color: white;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #e74c3c;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Title ------------------------- #
st.markdown("<h1 style='text-align: center;'>🔍 Ask GenAI Search Agent</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- Query Input ------------------------- #
query = st.text_input("Enter your query:", placeholder="e.g. What are the latest trends in Generative AI?")

# ------------------------- Search Button ------------------------- #
if st.button("Search"):
    if query.strip():
        with st.spinner("🔄 Searching for the latest info..."):
            response = get_search_results(query)

        # ------------------------- Display Results ------------------------- #
        st.success("✅ Here's what I found:")
        if response:
            # Format response into bullet points
            for line in response.split('.'):
                line = line.strip()
                if line:
                    st.markdown(f"- {line}.")
        else:
            st.warning("⚠️ No results found. Please try a different query.")
    else:
        st.warning("⚠️ Please enter a query before searching.")
