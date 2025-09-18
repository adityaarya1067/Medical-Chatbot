from pinecone import Pinecone

# Initialize Pinecone
pc=Pinecone(
    api_key="pcsk_GMTef_J1kiGrk8JFdKqUDH8H5f1zbwWdBYSocokLESNZ66PWumBfXeyrhqUJhCpyq3UCY"
)

# Connect to your index
index = pc.Index("medical-rag")
print(index.describe_index_stats())

dummy_vector = [0.0] * 768  # assuming 768-dimensional embeddings
response = index.query(vector=dummy_vector, top_k=5, include_metadata=True, namespace="chat-cache")

for match in response['matches']:
    print(f"ID: {match['id']}")
    print(f"Query: {match['metadata'].get('query')}")
    print(f"Response: {match['metadata'].get('response')}")
    print(f"Score: {match['score']}")
    print("="*50)
