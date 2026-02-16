import streamlit as st
from agents import SemanticSearchAgent

st.set_page_config(page_title="WBD Content Search", layout="wide")
st.title("🎬 WBD Content Intelligence POC")

@st.cache_resource
def load_agent():
    return SemanticSearchAgent()

agent = load_agent()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("View SQL"):
                st.code(msg["sql"], language="sql")

if prompt := st.chat_input("Ask about rights, deals, or financials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.answer(prompt)
            st.markdown(result["answer"])
            if "sql" in result:
                with st.expander("Generated SQL"):
                    st.code(result["sql"], language="sql")
            if "sources" in result:
                st.caption(f"Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sql": result.get("sql", "")
    })
