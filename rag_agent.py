import os
import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")


# State memory
class AgentState(TypedDict):
    question: str
    docs: list
    answer: str
    retry: int


# STEP 1: Retrieve
def retrieve(state):
    question = state["question"]

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    docs = results["documents"][0]

    return {
        "question": question,
        "docs": docs,
        "retry": state["retry"]
    }


# STEP 2: Generate
def generate(state):
    context = "\n".join(state["docs"])

    prompt = f"""
Use ONLY this context to answer.

Context:
{context}

Question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    return {
        "question": state["question"],
        "docs": state["docs"],
        "answer": response.content,
        "retry": state["retry"]
    }


# STEP 3: Grade answer
def grade(state):
    answer = state["answer"]

    if "don't know" in answer.lower() or len(answer) < 15:
        if state["retry"] < 1:
            return "retry"

    return "good"


# STEP 4: Rewrite query
def rewrite(state):
    new_question = "Explain clearly: " + state["question"]

    return {
        "question": new_question,
        "docs": [],
        "answer": "",
        "retry": state["retry"] + 1
    }


# Build Graph
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")

graph.add_conditional_edges(
    "generate",
    grade,
    {
        "good": END,
        "retry": "rewrite"
    }
)

graph.add_edge("rewrite", "retrieve")

app = graph.compile()


# Run
question = input("Ask Question: ")

result = app.invoke({
    "question": question,
    "docs": [],
    "answer": "",
    "retry": 0
})

print("\nFinal Answer:")
print(result["answer"])