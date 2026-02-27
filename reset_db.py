# reset_db.py
from qdrant_client import QdrantClient

# 連線到 Qdrant
client = QdrantClient("http://localhost:6333")

# 刪除舊的 collection
collection_name = "research_papers"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"✅ Collection '{collection_name}' has been deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")

print("Please restart your main app. It will recreate the collection with the correct dimension.")