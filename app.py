import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🚀 Dashboard")
    st.write("Upload PDFs")
    st.write("Ask Smart Questions")
    st.write("Powered by Groq + ChromaDB")

# Main Title
st.title("📄 PDF AI Chatbot")
st.caption("Upload your PDF and ask questions instantly.")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Question Input
question = st.text_input("Ask Question")

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Button
if st.button("Get Answer"):

    # Validation
    if uploaded_file is None:
        st.warning("Please upload a PDF.")
    
    elif question.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=250
        )

        docs = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Database
        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory="pdf_db"
        )

        # Retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        # Retrieve Relevant Chunks
        results = retriever.invoke(question)

        # Create Context
        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        # Show Retrieved Context
        st.write("## Retrieved Context")
        st.write(context)

        # Detect Summary-Type Questions
        question_lower = question.lower().strip()

        summary_triggers = [
            "what is the pdf about",
            "summarize",
            "summary",
            "tell me about",
            "give overview",
            "what does the pdf say",
        ]

        # Prompt Logic
        if any(trigger in question_lower for trigger in summary_triggers):

            prompt = f"""
You are a helpful PDF assistant.

Based only on the context below, give a short and clear summary of what the PDF is about.

You may infer the main topic from repeated themes in the context, but do not invent facts.

If the context is too weak to summarize confidently, say that clearly.

Context:
{context}

Question:
{question}

Answer:
"""

        else:

            prompt = f"""
You are a strict PDF question-answering assistant.

Use ONLY the context below.
Do not use outside knowledge.
Do not guess.

If the answer is not in the context, say:
"I could not find this information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

        # LLM Response
        response = llm.invoke(prompt)

        # Display Answer
        st.success("Answer generated successfully!")

        st.write(response.content)

# Footer
st.markdown("---")
st.caption("Built by BOSS 🚀")