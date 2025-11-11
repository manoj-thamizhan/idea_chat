# chat2.py
import streamlit as st

# --- Page setup ---
st.set_page_config(page_title="Critiq agent", page_icon="💡", layout="wide")

# --- Sidebar (UI only) ---
with st.sidebar:
    st.title("💡 Critiq Agent")
    st.button("🆕 New Chat")  # visual only
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    st.write("• Product Idea Exploration")
    st.write("• Market Fit Review")
    st.write("• UX Feedback Session")
    st.write("• AI Model Evaluation")
    st.markdown("---")

# --- Main chat area (your original code, unchanged) ---
st.title("Critiq agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None  # placeholder, no functionality

agent = st.session_state.agent

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your product idea or ask for an analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking like a product-market analyst..."):
        # no functionality
        reply = "_(This is a mock UI — no backend connected.)_"

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
