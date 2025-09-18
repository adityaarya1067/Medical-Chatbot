from pinecone import Pinecone

# Initialize Pinecone
pc=Pinecone(
    api_key="pcsk_GMTef_J1kiGrk8JFdKqUDH8H5f1zbwWdBYSocokLESNZ66PWumBfXeyrhqUJhCpyq3UCY"
)

# Connect to your index
index = pc.Index("medical-rag")

# Try fetching up to 10 vectors from "chat_context" namespace without any filters
# Replace "s2" with the actual session_id you saw above
res_filtered = index.query(
    vector=[0.0] * 768,
    namespace="chat_context",
    top_k=10,
    filter={
        "session_id": {"$eq": "s2"},
        "source": {"$eq": "chat_history"}
    },
    include_metadata=True
)

matches = res_filtered.get("matches", [])
print(f"[DEBUG] Filtered query returned {len(matches)} results")

for match in matches:
    print("Session ID:", match["metadata"].get("session_id"))
    print("Text preview:", match["metadata"].get("text", "")[:100])


