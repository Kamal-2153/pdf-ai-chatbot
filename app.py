import streamlit as st

st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1 {
    color: #00FFD1;
}
.stButton button {
    background-color: #00FFD1;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stTextInput input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🚀 Dashboard")
    st.write("Upload PDFs")
    st.write("Ask Smart Questions")
    st.write("Powered by Groq + ChromaDB")

# Main UI
st.title("📄 PDF AI Chatbot")
st.caption("Upload your PDF and ask questions instantly.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

question = st.text_input("Ask Question")

if st.button("Get Answer"):
    if uploaded_file and question:
        st.success("Answer generated successfully!")
        st.write("This is where your AI answer will appear.")
    else:
        st.warning("Please upload file and ask question.")

st.markdown("---")
st.caption("Built by BOSS 🚀")