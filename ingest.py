import chromadb
from pathlib import Path

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")

docs_path = Path("docs")

documents = []
ids = []

count = 0

for file in docs_path.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append(text)
    ids.append(str(count))
    count += 1

collection.add(documents=documents, ids=ids)

print("Documents stored successfully.")