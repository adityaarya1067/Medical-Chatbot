import json
from tqdm import tqdm
from pinecone import Pinecone

pc=Pinecone(
    api_key="pcsk_GMTef_J1kiGrk8JFdKqUDH8H5f1zbwWdBYSocokLESNZ66PWumBfXeyrhqUJhCpyq3UCY"
)

# Connect to your index
index = pc.Index("medical-rag")
# Constants
NAMESPACE = "rag_docs"
OUTFILE = "stored_pmc_ids.json"

print("[INFO] Initializing Pinecone client...")

def export_pmc_ids(index):
    print("[INFO] Fetching vector IDs from namespace:", NAMESPACE)

    try:
        all_ids = index.describe_index_stats()["namespaces"].get(NAMESPACE, {}).get("vector_count", 0)
        if all_ids == 0:
            print("[INFO] No vectors found in namespace.")
            return
    except Exception as e:
        print("[ERROR] Failed to describe index stats:", e)
        return

    # Since Pinecone doesn't expose a method to get all vector IDs directly,
    # assume you already have them cached, or re-use them from a previously stored list.
    # For demo purposes, we simulate you already know your vector IDs.

    print("[WARNING] Pinecone doesn't support listing vector IDs directly via Python SDK.")
    print("You need to have previously stored your vector IDs.")

    # --------- Replace this part if you already have a list of vector IDs ----------
    # E.g., from a previous map file or export
    with open("downloaded_map.json") as f:
        id_map = json.load(f)
        all_ids = list(id_map.keys())
    # -------------------------------------------------------------------------------

    print(f"[INFO] Found {len(all_ids)} vector IDs. Fetching metadata...")

    pmc_ids = set()
    batch_size = 100

    for i in tqdm(range(0, len(all_ids), batch_size)):
        batch = all_ids[i:i+batch_size]
        try:
            response = index.fetch(ids=batch, namespace=NAMESPACE)
        except Exception as e:
            print(f"[ERROR] Fetch failed for batch {i}: {e}")
            continue

        for vec in response.vectors.values():
            metadata = vec.metadata or {}
            pmc_id = metadata.get("pmc_id", "").strip()
            if pmc_id:
                pmc_ids.add(pmc_id)

    with open(OUTFILE, "w") as f:
        json.dump(sorted(pmc_ids), f, indent=2)

    print(f"[DONE] Extracted {len(pmc_ids)} unique pmc_ids to '{OUTFILE}'")

export_pmc_ids(index)  