import chromadb

# Create or connect to your ChromaDB instance
client = chromadb.Client()

# Replace 'your_cache_collection_name' with your actual collection name
collection = client.get_or_create_collection(name="chat-cache")

# Delete the collection
client.delete_collection(name=collection.name)

print(f"Collection '{collection.name}' deleted successfully.")
