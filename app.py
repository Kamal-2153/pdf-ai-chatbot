import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

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

        # Save uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        # Embeddings
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ChromaDB
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            docs,
            embeddings
        )

        retriever = vectorstore.as_retriever()

        # Retrieve relevant docs
        results = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in results])

        # Prompt
        prompt = f"""
Answer only using this context:

{context}

Question: {question}
"""

        # LLM Response
        response = llm.invoke(prompt)

        st.success("Answer generated successfully!")

        st.write(response.content)

    else:
        st.warning("Please upload file and ask question.")

st.markdown("---")
st.caption("Built by BOSS 🚀")