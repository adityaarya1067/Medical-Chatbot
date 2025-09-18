import os
from dotenv import load_dotenv
load_dotenv()
import json
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import redis
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb import Client
from typing import Optional, List,Tuple,Dict,Any
import uuid
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder
import pandas as pd
import re
import ollama
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
from datetime import datetime,timezone
from transformers import pipeline
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langdetect import detect
import fitz  # PyMuPDF
import os
import asyncio
from langchain_pinecone import PineconeVectorStore
# from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.documents import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
# from ragas import SingleTurnSample
# from ragas.metrics import AspectCritic
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from biobert import BioBERTLangChainEmbedder
import torch.nn.functional as F



OPEN


# evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc=Pinecone(
    api_key=api_key
)
# Connect to your index
index = pc.Index("medical-rag")

# print(index.describe_index_stats())
# ========== CONFIG ============
# Ensure NLTK data is downloaded
nltk_data_path = 'nltk_data' 
# Download required NLTK data
nltk.download('punkt')    
nltk.download('stopwords')
nltk.data.path.append(nltk_data_path)

# Download the required NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)


# ========== CONFIG ============
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 
device = "cuda" if torch.cuda.is_available() else "cpu"

AZURE_CONTAINER = "pmc-fulltext"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "medical_docs"
DOWNLOADED_MAP_PATH = "downloaded_map.json"
NON_ENGLISH_MAP_FILE = "non_english_seen_map.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP =100
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)
SIMILARITY_THRESHOLD = 0.98
CACHE_DIR = "local_cache"
EMBED_FILE = os.path.join(CACHE_DIR, "embeddings.pkl")
CACHE_FILE = os.path.join(CACHE_DIR, "cache.json")  
SIMILARITY_THRESHOLD = 0.98
embedding_model = BioBERTLangChainEmbedder(biobert_tokenizer, biobert_model, device=device)
# vectorstore = Pinecone(index=index, embedding=embedding_model, namespace="rag_docs")
##############-----CONTEXT AWARE---##########
### ✅ Utility: Parse user/assistant pairs from raw context
def parse_session_context(session_context):
    parsed = []
    for entry in session_context:
        try:
            user_part = entry.split("[User]:")[1].split("[Assistant]:")[0].strip()
            assistant_part = entry.split("[Assistant]:")[1].strip()
            parsed.append((user_part, assistant_part))
        except Exception:
            continue
    return parsed

### ✅ Prompt-based summarizer using any LLM (e.g., LLaMA 3.1)
def summarize_with_llm(text, llm_generate_fn, prompt_template=None):
    prompt_template = prompt_template or (
        "You are a biomedical assistant. Summarize the following chat history between a user and an assistant.\n\n"
        "===\n\n{chat}\n\n===\n\nSummarize the above into a concise, medically accurate recap. Limit to key facts and questions.\n\nSummary:"
    )
    full_prompt = prompt_template.format(chat=text)
    return llm_generate_fn(full_prompt)


### ✅ Duplicate check in Pinecone (same session_id and text)
def is_already_embedded(index, session_id, text):
    embedded = embed_texts([text], biobert_tokenizer, biobert_model, device=device)[0]
    query_results = index.query(
        vector=embedded,
        top_k=3,
        include_metadata=True,
        namespace="chat_context",
        filter={"session_id": {"$eq": session_id}},
    )
    for match in query_results.get("matches", []):
        if match.get("metadata", {}).get("text", "") == text:
            return True
    return False


### ✅ Actual LLM generator
def generate_answer_with_llm(prompt):
    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "You are a professional biomedical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return "I'm sorry, I couldn't generate a response at the moment."

def needs_full_context(query: str) -> bool:
    """
    Heuristic check to decide if a query needs the full chat history.
    Can be replaced with a classifier or semantic check later.
    """
    triggers = [
        "based on our conversation", "as we discussed", "summarize", "what have we talked",
        "analyze all", "recap", "everything so far", "overall", "entire session"
    ]
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in triggers)

 
### ✅ Store query/response pair in Pinecone

def store_chat_history(index, session_id, query, answer, embed_texts=None, tokenizer=None, model=None, device="cpu"):
    """
    Stores user query and answer into Pinecone under a chat_context namespace.
    Requires embedding if you plan to filter by dummy vector later.
    """
    print(f"[STORE] Storing chat history for session: {session_id}")

    if embed_texts is None or tokenizer is None or model is None:
        raise ValueError("Embedding function and model/tokenizer must be provided.")

    text = f"User: {query}\nAssistant: {answer}"
    embedding = embed_texts([text], tokenizer, model, device=device)[0]

    timestamp = datetime.utcnow().isoformat()
    token_count = len(text.split())

    vector = {
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "text": text,
            "token_count": token_count,
            "timestamp": timestamp,
            "source": "chat_history"
        }
    }

    index.upsert(vectors=[vector], namespace="chat_context")

### ✅ Retrieve recent + summarized past context
def retrieve_chat_history(index, session_id, latest_query, embed_texts, tokenizer, model, device="cpu",
                          token_limit=1024, summary_token_threshold=800, recent_k=5):
    """
    Retrieves chat history with logic to return only last K unless full history is requested.
    """
    print(f"[DEBUG] Retrieving chat history for session: {session_id}")

    # Embed the current query
    try:
        query_text = f"User: {latest_query}"
        embedding = embed_texts([query_text], tokenizer, model, device=device)[0]
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return ""

    try:
        response = index.query(
            vector=embedding,
            namespace="chat_context",
            top_k=50,  # Get enough to allow summarization if needed
            filter={
                "session_id": {"$eq": session_id},
                "source": {"$eq": "chat_history"}
            },
            include_metadata=True
        )
    except Exception as e:
        print(f"[ERROR] Pinecone query failed: {e}")
        return ""

    matches = response.get("matches", [])
    matches = sorted(matches, key=lambda x: x["metadata"].get("timestamp", ""))

    if not matches: 
        print("[DEBUG] No chat history found.")
        return ""

    # If query requires full context (or summarization)
    if needs_full_context(latest_query):
        docs = [m["metadata"]["text"] for m in matches]
        all_tokens = [m["metadata"].get("token_count", 0) for m in matches]
        total_tokens = sum(all_tokens)

        if total_tokens <= token_limit:
            print("[DEBUG] Returning full session context.")
            return "\n".join(docs)

        # Split into old vs recent for summarization
        old_docs, recent_docs = [], []
        token_count = 0

        for doc, tok in zip(reversed(docs), reversed(all_tokens)):
            if token_count + tok <= summary_token_threshold:
                recent_docs.insert(0, doc)
                token_count += tok
            else:
                old_docs.insert(0, doc)

        try:
            summary_text = summarize_with_llm(" ".join(old_docs), generate_answer_with_llm)
            print("[DEBUG] Summary of old session history generated.")
        except Exception as e:
            print(f"[WARN] Summarization failed: {e}")
            summary_text = "[Summary unavailable due to error.]"

        return summary_text + "\n\n" + "\n".join(recent_docs)

    else:
        # ✅ Only return last K responses
        last_k = matches[-recent_k:]
        print(f"[DEBUG] Returning last {recent_k} context entries.")
        return "\n".join(m["metadata"]["text"] for m in last_k)

##############------CACHE-----###############

# def tensor_to_list(embedding):
#     if isinstance(embedding, torch.Tensor):
#         return embedding.detach().cpu().tolist()
#     elif isinstance(embedding, list) and isinstance(embedding[0], torch.Tensor):
#         return [e.detach().cpu().tolist() for e in embedding]
#     return embedding


# def search_cached_response(index, query):
#     print("[CACHE] Searching cache for query:", query)

#     embedding = embed_texts([query], biobert_tokenizer, biobert_model, device=device)[0]

#     if not embedding or not isinstance(embedding, list):
#         print("[CACHE] Error: Invalid or missing embedding.")
#         return None
#     try:
#         results = index.query(
#             vector=embedding,
#             top_k=3,
#             include_metadata=True,
#             namespace="cache"
#         )
#     except Exception as e:
#         print("[CACHE] Error querying Pinecone:", str(e))
#         return None

#     matches = results.get("matches", [])

#     if not matches:
#         print("[CACHE] No cached results found.")
#         return None

#     for match in matches:
#         try:
#             similarity = match.get("score", 0)
#             print(f"[CACHE] Found result with similarity: {similarity:.4f}")
#             if similarity >= SIMILARITY_THRESHOLD:
#                 return match.get("metadata", {}).get("response")
#         except Exception as e:
#             print("[CACHE] Error processing match:", str(e))
#             continue

#     print("[CACHE] No similar enough cached result found.")
#     return None

# def cache_response(index, query, answer):
#     print("[CACHE] Caching response for query:", query)

#     embedding = embed_texts([query], biobert_tokenizer, biobert_model, device=device)[0]

#     vector = {
#         "id": str(uuid.uuid4()),
#         "values": embedding,
#         "metadata": {
#             "query": query,
#             "response": answer,
#             "source": "cache"
#         }
#     }

#     index.upsert(vectors=[vector], namespace="cache")

#     print("[CACHE] Cached query-response pair successfully.")

def ensure_cache_files():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "wb") as f:
            pickle.dump({}, f)
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)

def load_embeddings():
    with open(EMBED_FILE, "rb") as f:
        return pickle.load(f)


def save_embeddings(data):
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(data, f)
        
def load_cache():
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def save_cache(data):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def search_cached_response_local(query, embed_texts, tokenizer, model, device):
    # print("[CACHE] Searching local cache for query:", query)
    ensure_cache_files()

    query_embedding = embed_texts([query], tokenizer, model, device=device)[0]

    if not query_embedding:
        print("[CACHE] Error: Invalid or missing embedding.")
        return None

    embeddings = load_embeddings()
    cache = load_cache()

    if not embeddings:
        print("[CACHE] No local embeddings found.")
        return None

    all_embeddings = list(embeddings.values())
    all_keys = list(embeddings.keys())

    similarity_scores = cosine_similarity(
        [query_embedding],
        all_embeddings
    )[0]

    for i, score in enumerate(similarity_scores):
        print(f"[CACHE] Similarity with cached query '{all_keys[i]}': {score:.4f}")
        if score >= SIMILARITY_THRESHOLD:
            return cache.get(all_keys[i])  # response

    print("[CACHE] No similar enough cached result found.")
    return None

def cache_response_local(query, response, embed_texts, tokenizer, model, device):
    print("[CACHE] Caching response locally for query:", query)
    ensure_cache_files()

    embedding = embed_texts([query], tokenizer, model, device=device)[0]

    cache = load_cache()
    embeddings = load_embeddings()

    cache[query] = response
    embeddings[query] = embedding

    save_cache(cache)
    save_embeddings(embeddings)

    print("[CACHE] Cached locally successfully.")

# ========== UTILS ==============

fallback_phrases = [
    "not contain enough information",
    "cannot answer your question",
    "insufficient context",
    "please upload more",
    "unclear from the context",
    "no information provided",
    "no relevant data found",
    "no available context", 
    "I'm unable to provide an answer",
    "I cannot provide medical advice",
    "does not explicitly mention",
    "information is indirect and may not be directly relevant",
    "further information is needed",
    "documents do not address",
    "answer based on the available information",
    "unable to locate sufficient detail",
    "no direct reference to",
    "context does not support a conclusive answer",
    "no specific data available",
    "recommend reviewing additional documents",
    "data is inconclusive",
    "no definitive statement can be made",
    "requires more targeted information",
    "details are not discussed in the context provided",
    "this is not mentioned in the provided excerpts",
    "context is missing key details",
    "could not verify from context",
    "topic is not covered in the documents",
    "question cannot be answered definitively",
    "answer may be incomplete due to lack of information"
    "According to your query, context is insufficient.",
    "However, the document does not provide information",
    "are not clearly mentioned",
    "context is insufficient",
    "The context does not provide",
    "is not mentioned",
    "cannot answer",
    "I do not have more information",
    "data do not provide any evidence",
    "There is no specific information"

]

def is_fallback_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in fallback_phrases)

def normalize_embedding(embedding):
    arr = np.array(embedding)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm != 0 else arr.tolist()

def load_keywords_from_file(path='keywords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set(kw.strip().lower() for kw in f if kw.strip())

def extract_doi(text):
    doi_pattern = r"(?:doi:\s*)?(10\.\d{4,9}/[^\s]+)"
    match = re.search(doi_pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_pmc_id_from_filename(filename):
    """Extracts the PMC ID (e.g., PMC6810720) from a given filename."""
    match = re.search(r"(PMC\d+)", filename)
    return match.group(1) if match else ""

def extract_keywords(text):
    keyword_pattern = r"(?i)keywords\s*:\s*([^\n\r]+)"
    match = re.search(keyword_pattern, text)
    if match:
        return [kw.strip().lower() for kw in match.group(1).split(",")]
    return []

def extract_corresponding_author(text):
    """
    Extracts the corresponding author's name and email address from the text.
    Supports both explicit 'Corresponding author' labels and implicit patterns.
    Returns a dictionary {'name': ..., 'email': ...} or None if not found.
    """
    text = text.replace("\n", " ")

    # Pattern 1: Explicit "corresponding author"
    pattern_explicit = r"(?i)corresponding author\s*[:\-]?\s*([\w\s\.\-Á-ÿ]+?),[^@]*?([\w\.-]+@[\w\.-]+)"
    match = re.search(pattern_explicit, text)
    if match:
        return {"name": match.group(1).strip(), "email": match.group(2).strip()}

    # Pattern 2: Implicit - Name followed by email (e.g., "John Smith - john@email.com")
    pattern_implicit = r"([A-Z][a-zA-Z\.\'\-Á-ÿ]+(?:\s+[A-Z][a-zA-Z\.\'\-Á-ÿ]+)+)\s*[-–—:]*\s*([\w\.-]+@[\w\.-]+)"
    match = re.search(pattern_implicit, text)
    if match:
        return {"name": match.group(1).strip(), "email": match.group(2).strip()}

    return None

def is_relevant(text, keywords):
    return any(keyword.lower() in text.lower() for keyword in keywords)

def load_downloaded_map():
    if os.path.exists(DOWNLOADED_MAP_PATH):
        with open(DOWNLOADED_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

def load_skipped_map():
    if os.path.exists(SKIPPED_MAP_PATH):  
        with open(SKIPPED_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

def load_non_english_map():
    if os.path.exists(NON_ENGLISH_MAP_FILE):
        with open(NON_ENGLISH_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

SKIPPED_MAP_PATH = "skipped_downloaded_map.json"

def save_downloaded_map(downloaded_map):
    with open(DOWNLOADED_MAP_PATH, 'w') as f:
        json.dump(downloaded_map, f)

def save_skipped_map(skipped_map):
    with open(SKIPPED_MAP_PATH, 'w') as f:
        json.dump(skipped_map, f)


def save_non_english_map(non_english_map):
    with open(NON_ENGLISH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(non_english_map, f)


# ========== 1. DATA INGESTION ==========
def extract_text_from_bytes(pdf_bytes):
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
    except:
        pass
    return text.strip()

def download_pdfs_from_blob(max_pdfs=100):
    print("[INFO] Downloading PDFs from Azure Blob Storage...")

    downloaded_map = load_downloaded_map()
    skipped_downloaded_map = load_skipped_map()
    non_english_map = load_non_english_map()  # <--- new

    blob_service = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
    container_client = blob_service.get_container_client(AZURE_CONTAINER)

    pdfs = []
    count = 0
    
    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            continue
        if downloaded_map.get(blob.name, False):
            continue
        if skipped_downloaded_map.get(blob.name, False):
            continue
        if non_english_map.get(blob.name, False):  
            continue

        blob_client = container_client.get_blob_client(blob)

        try:
            pdf_bytes = blob_client.download_blob(timeout=300).readall()
            if not pdf_bytes or len(pdf_bytes) < 100:
                print(f"[SKIPPED] {blob.name} - file too small")
                skipped_downloaded_map[blob.name] = True
                continue

            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)

            if num_pages <= 2:
                print(f"[SKIPPED] {blob.name} - only {num_pages} pages")
                skipped_downloaded_map[blob.name] = True
                continue

            # Extract text for language detection
            extracted_text = extract_text_from_bytes(pdf_bytes)
            if not extracted_text or len(extracted_text) < 50:
                print(f"[SKIPPED] {blob.name} - text too short or empty")
                skipped_downloaded_map[blob.name] = True
                continue

            try:
                lang = detect(extracted_text)
            except:
                lang = "undetected"

            if lang != "en":
                print(f"[NON-ENGLISH] {blob.name} detected as '{lang}' → Skipping")
                non_english_map[blob.name] = True
                continue

            # If everything is fine, add to final list
            print(f"[OK] Downloaded & accepted: {blob.name} ({num_pages} pages, lang={lang})")
            pdfs.append((blob.name, pdf_bytes))
            downloaded_map[blob.name] = True
            count += 1

        except PdfReadError:
            print(f"[ERROR] Invalid PDF: {blob.name}")
            skipped_downloaded_map[blob.name] = True

        except Exception as e:
            print(f"[ERROR] Failed to process {blob.name}: {e}")
            skipped_downloaded_map[blob.name] = True

        if count >= max_pdfs:
            break

    save_downloaded_map(downloaded_map)
    save_skipped_map(skipped_downloaded_map)
    save_non_english_map(non_english_map)  # <-- new
    print(f"[INFO] Total valid English PDFs downloaded: {len(pdfs)}")
    return pdfs

# ========== 2. TEXT EXTRACTION AND CHUNKING ==========

def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "" 
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,            
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
 
# ========== 3. EMBEDDING WITH BIOBERT ==========
def embed_texts(texts, tokenizer, model, device):
    print(f"[INFO] Generating embeddings for {len(texts)} chunks using BioBERT...")

    model = model.to(device)
    model.eval()
    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            embeddings.append(mean_embedding.squeeze().cpu().tolist())
    return embeddings  


# ========== 4. STORE TO PINECONE ==========

MAX_PAYLOAD_SIZE = 3*1024*1024

def estimate_vector_size(vector):
    """
    Estimate the size of a vector in bytes.
    Assuming embedding is a list of floats.
    """
    return len(json.dumps(vector).encode('utf-8')) + 64  # 64 bytes for metadata overhead
 
def store_embeddings_pinecone(index, chunks, embeddings, full_doc_text, filename):
    print("[INFO] Storing embeddings in Pinecone with metadata...")

    # Extract metadata
    doi = extract_doi(full_doc_text)
    pmc_id = extract_pmc_id_from_filename(os.path.basename(filename))
    added = 0
    skipped = 0

    upsert_items = []
    current_batch_size = 0
    def flush_batch(batch):
        if batch:
            print(f"[INFO] Storing {len(batch)} chunks in Pinecone...")
            index.upsert(vectors=batch, namespace="rag_docs")
            

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        

        chunk_id = f"doc_{hash(full_doc_text)}_chunk_{i}_{uuid.uuid4().hex}"

        metadata = {
            "text": chunk,
            "pmc_id": pmc_id or "",
            "doi": doi or "",
            "chunk_index": i
        }
        vector = {
            "id": chunk_id,
            "values": embedding,  # Ensure normalized embedding
            "metadata": metadata
        }
    
        vector_size = estimate_vector_size(embedding)

        if current_batch_size + vector_size > MAX_PAYLOAD_SIZE:
            flush_batch(upsert_items)
            upsert_items = []
            current_batch_size = 0

        upsert_items.append(vector)
        current_batch_size += vector_size
        added += 1

        chunk_words = len(chunk.split())
        print(f"[DEBUG] Chunk {i+1}: {chunk_words} words")
        print(f"[DEBUG] Preview: {chunk[:100].strip().replace('/n', ' ')}...\n")
    
    flush_batch(upsert_items)

    
    print(f"[INFO] Storing {added} chunks in Pinecone...")
      
    return index


# ========== 5. RETRIEVE RELEVANT CHUNKS ============

def retrieve_chunks(query, index, tokenizer, model, device, top_k=20, similarity_threshold=0.2):
    # Step 1: Generate embedding for the query using bi-encoder
    query_embedding = embed_texts([query], tokenizer, model, device=device)[0]

    # Step 2: Query Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,  
        include_metadata=True,
        namespace="rag_docs"
    )

    # Step 3: Filter results with a similarity threshold
    filtered_matches = [
        {
            "score": match["score"],

            "text": match.get("metadata", {}).get("text", ""),
            "metadata": match.get("metadata", {}),

        }
        for match in results["matches"]
        
        if match["score"] >= (1 - similarity_threshold)
    ]
    
    # Sort matches by similarity score
    filtered_matches.sort(key=lambda x: x["score"], reverse=True)

    # Return embeddings and top matches
    return query_embedding, filtered_matches[:top_k]

# from langchain.retrievers import VectorStoreRetriever
# from langchain_pinecone import PineconeVectorStore

# document_content_description = "Medical documents containing clinical trial protocols, drug labels, regulatory or guideline text, and research excerpts."
# metadata_field_info = [
#     AttributeInfo(name="pmc_id", description="The PubMed Central ID of the document.",type="string"),
#     AttributeInfo(name="doi", description="The Digital Object Identifier of the document.",type="string")
# ]

# vectorstore = PineconeVectorStore.from_existing_index(
#     index_name="medical-rag",
#     embedding=embedding_model,
#     namespace="rag_docs"
# )
# # retriever = vectorstore.as_retriever()
# retriever = SelfQueryRetriever.from_llm(
#     llm, vectorstore, document_content_description, metadata_field_info, verbose=True
# )
# # retriever = vectorstore.as_retriever() 
# docs = vectorstore.similarity_search("cancer", k=2)
# print(docs[0].metadata if docs else "No documents found")
# # doc = vectorstore.similarity _search("cancer", k=1)[0]
# # print(doc.metadata) 
# def retrieve_chunks_langchain(query: str):
#     docs = retriever.invoke(query)
#     print(f"[DEBUG] Retrieved {len(docs)} documents for query: {query}")
#     if not docs:   
#         print("[DEBUG] No relevant documents found.")
#         return []
#     results = []
#     for doc in docs:
#         results.append({
#             "text": doc.page_content,
#             "metadata": doc.metadata,
#         })
#     return results
# ========== 6. GENERATE ANSWER WITH LLaMA 3 ==========

def generate_answer(context_chunks, query, session_context=None):
    context = "\n\n".join(
    f"{chunk['text']}\n[DOI: {chunk['metadata'].get('doi', 'N/A')}, PMCID: {chunk['metadata'].get('pmc_id', 'N/A')}]"
    for chunk in context_chunks
)
    print(f"Query is: {query.upper()}")


    prompt = f"""You are a highly reliable and professional assistant with extensive knowledge of medical and pharmaceutical content. Your task is to answer questions accurately and clearly using context extracted from medical PDFs that have been embedded in a vector database.
 
You will receive:
- A **user query**
- A **set of context excerpts** retrieved from medical PDF documents, Along with these excerpts, **metadata will be provided**
 
INSTRUCTIONS
────────────
1. USE ONLY THE SUPPLIED CONTEXT  
   • Base every statement solely on the documents provided (e.g., clinical‑trial protocols, drug labels, regulatory or guideline text, research excerpts).  
   • Do NOT add external medical knowledge, assumptions, or prior training data.  
2. PRESENT FINDINGS PLAINLY  
   • State conclusions in clear prose.  
   • NEVER mention or hint at tables, figures, images, graphs, panels, charts, plots, or supplemental material.  
   • Forbidden tokens/phrases (case‑insensitive, punctuation/whitespace ignored):  
     fig, figure, table, image, panel, chart, graph, plot, supplementary, supplemental, dataset S, fig S, figure S, etc.  
   • If evidence exists only inside such visuals and is not described textually, treat it as unavailable (see Rule 7).  
3. NO SPECULATION  
   • If the context does not clearly justify a statement, omit it.  
4. PROFESSIONAL TONE  
   • Write concisely and accurately for clinical/regulatory audiences. Omit conversational filler.  
5. CITATIONS  
   • Support each substantive claim with the document’s metadata, placed in parentheses at the end of the sentence or paragraph.  
   • Accepted formats (use what is available):  
     – DOI: 10.1039/c8sc00367j  
     – PMCID: PMC2136085  
     – FDA Label: [Drug Name, 2024‑05‑10]  
     – EMA SmPC: [Product Name, 2023‑11‑30]  
     – Guideline: [Organization, 2022]  
   • Do NOT include URLs unless explicitly requested.  
   • NEVER use wording such as “According to Table X…”.  
6. MULTIPLE DOCUMENTS  
   • When several sources address the topic, synthesise them.  
   • If sources conflict, present each viewpoint with its citation and note the discrepancy.  
7. INSUFFICIENT CONTEXT  
   • If any required detail is missing, unclear, appears only in a forbidden visual, or cannot be supported unequivocally by the text, reply EXACTLY with:  
     "According to your query, context is insufficient."  
   • Return this sentence by itself—no additional text, reasoning, or citations like "However, the document does not provide information on the materials these bellyboards are made of. It only mentions that temperature changes during routine clinical examinations with these devices were investigated."
   are not required for these mention the above sentence.
   
8. FINAL COMPLIANCE CHECK  
   • Before sending, ensure:  
     – Every claim is supported by text and properly cited.  
     – No forbidden tokens/phrases remain.  
     – If Rule 7 applies, the response contains ONLY the prescribed insufficiency statement.  
    
IMPORTANT NOTES
───────────────
• Always follow Rule 7 verbatim when necessary.  
• Perform a final scan for forbidden tokens/phrases before replying.  
• Maintain a concise, professional tone throughout.  
 
EXAMPLES
────────
• **Sufficient context**  
  Efficient cargo delivery requires a zipper‑like hybridisation design; anti‑zipper designs were ineffective (DOI: 10.1039/c8sc00367j).  
• **Insufficient context**  
  According to your query, context is insufficient.
 
 
Your responses should always focus on **evidence-based knowledge** from the provided documents and should help the user make informed medical decisions or understand regulatory requirements, clinical trial data, or pharmaceutical information
 
Use the following conversation history and document context to answer the user's new question.
 
Conversation History:
{session_context}
 
Document Context:
{context}
 
User Question:
{query}
Answer:"""
    
    messages = [
        {"role": "system", "content": "You are a medical assistant. Be precise, avoid speculation, and cite only the provided context."},
        {"role": "user", "content":   prompt}
    ]

    response = ollama.chat(
        model="llama3.1:8b",
        messages=messages
    )
    return response['message']['content'].strip()
      
# ========== 7. PREPROCESS QUERY ==========
def preprocess(text):
    """
    Preprocess the input text by tokenizing, converting to lowercase,
    removing stop words and non-alphanumeric tokens.
    """
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Define stop words
    stop_words = set(stopwords.words('english'))
    # Filter tokens
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # remove generic words
    toremove = {
        'mg', 'ml', 'oral', 'tablet', 'tablets', 'capsule', 'capsules', 'solution', 'suspension',
        'injection', 'injections', 'inhalation', 'inhaler', 'inhalers', 'drug', 'drugs', 'medication',
        'disease', 'diseases','{','}'
        'medications', 'medicine', 'medicines', 'treatment', 'treatments', 'therapy', 'therapies',
        'dose', 'doses', 'dosage', 'dosages', 'administration', 'acid', 'acids', 'documents',
        'document', 'information', 'patient', 'patients', 'report', 'reports', 'study', 'studies',
        'result', 'results', 'data', 'file', 'files', 'details', 'description', 'content',
        'publication', 'publications', 'article', 'articles', 'reference', 'references', 'content',
        'version', 'versions', 'test', 'tests', 'trial', 'trials', 'method', 'methods', 'system',
        'systems', 'evaluation', 'evaluations', 'protocol', 'protocols', 'dose', 'level', 'levels',
        'standard', 'standards', 'procedures', 'review', 'reviews', 'procedure', 'objectives',
        'objective', 'category', 'categories', 'class', 'classes', 'group', 'groups', 'show',
       
        # Add more generic words to remove
        'regarding', 'about', 'concerning', 'related', 'associated', 'in', 'on', 'by', 'of',
        'with', 'to', 'from', 'as', 'for', 'and', 'the', 'a', 'an', 'that', 'this', 'these',
        'those', 'which', 'such', 'any', 'all', 'each', 'many', 'much', 'other', 'others',
        'more', 'few', 'overview', 'insights', 'analysis', 'guidelines', 'effects', 'factors',
        'methods', 'results', 'research', 'data', 'researchers', 'patients', 'trials',
        'studies', 'criteria', 'conclusions', 'summary', 'implications', 'applications',
        'results', 'evidence', 'findings', 'presentation', 'exploration',
        
        # Expanded terms
        'chemical', 'chemicals', 'compound', 'compounds', 'substance', 'substances',
        'formula', 'formulas', 'properties', 'property', 'characteristics',
        'characteristic', 'composition', 'components', 'component', 'ingredients',
        'ingredient', 'mechanism', 'mechanisms', 'action', 'actions', 'impact',
        'effects', 'administration', 'route', 'routes', 'indication', 'indications',
        'usage', 'usage', 'use', 'uses', 'recommendations', 'recommendation',
        'safety', 'safeness', 'risk', 'risks', 'adverse', 'reactions', 'reaction',
        'efficacy', 'efficacious', 'benefit', 'benefits', 'review', 'evaluation',
        'evaluation', 'implication', 'implications', 'impact', 'clinical', 'clinical trials',
        'clinical data', 'guidance', 'findings', 'treatment', 'treatment options', 'pubmed','data','title',       
        # Miscellaneous terms
        'criteria', 'finding', 'suggestions', 'suggestion', 'analyses', 'analysis',
        'clinical', 'clinical findings', 'support', 'supports', 'considerations',
        'consideration', 'context', 'situations', 'case', 'cases', 'history',
        'pathway', 'pathways', 'evaluation', 'measure', 'measurement', 'recommend',
        'effectiveness', 'guide', 'reporting', 'results', 'summary', 'future', 'studies',
       
        # General filler words
        'also', 'but', 'or', 'if', 'then', 'however', 'meanwhile', 'during', 'want',
        'after', 'before', 'although', 'while', 'despite', 'since', 'unless',
        'yet', 'so', 'therefore', 'thus', 'hence', 'but', 'although', 'where',
        'when', 'which', 'that', 'is', 'are', 'was', 'were', 'be', 'been',
       
        'advantage', 'advantages', 'disadvantage', 'disadvantages', 'pros', 'cons',
        'benefit', 'benefits', 'drawback', 'drawbacks', 'merit', 'merits', 'downside',
        'downsides', 'upside', 'upsides', 'positive', 'negatives', 'negative', 'impact',
        'outcome', 'outcomes', 'effect', 'effects',
    }
   
    # Remove unwanted words
    filtered = [word for word in filtered if word not in toremove]
    # Remove duplicates and sort
    return set(filtered)

# ========== 8. Re-ranking relevant chunks ==========
from typing import List, Dict

# def rerank_chunks(query: str, retrieved_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
#     """
#     Reranks retrieved chunks using a cross-encoder model based on relevance to the query.

#     Args:
#         query (str): The user's question/query.
#         retrieved_chunks (List[Dict]): A list of dicts, each with 'text' and 'metadata' keys.
#         top_k (int): Number of top chunks to return after reranking.

#     Returns:
#         List[Dict]: Top-k reranked chunks with cross-encoder scores.
#     """
#     if not retrieved_chunks:
#         print("[INFO] No retrieved chunks to rerank.")
#         return []
#     # Step 1: Form query-chunk pairs
#     pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]

#     # Step 2: Score them using the cross-encoder
#     scores = cross_encoder.predict(pairs)

#     # Step 3: Attach scores to chunks
#     reranked = []
#     for i, chunk in enumerate(retrieved_chunks):
#         reranked.append({
#             "text": chunk["text"],
#             "metadata": chunk["metadata"],
#             "cross_score": float(scores[i])  # ensure float for JSON serializability
#         })

#     # Step 4: Sort chunks by score descending
#     reranked.sort(key=lambda x: x["cross_score"], reverse=True)

#     # Step 5: Return top-k
#     return reranked[:top_k]
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def rerank_chunks(query, relevant_chunks, cross_encoder_model, cross_encoder_tokenizer, device):
    # Prepare query-passage pairs
    pairs = [(query, chunk["text"]) for chunk in relevant_chunks]

    # Tokenize all pairs
    inputs = cross_encoder_tokenizer(
        [p[0] for p in pairs],  # queries
        [p[1] for p in pairs],  # passages
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Compute scores
    with torch.no_grad():
        logits = cross_encoder_model(**inputs).logits.squeeze()  # shape: [N]

    # Attach scores back to chunks
    reranked = []
    for chunk, score in zip(relevant_chunks, logits):
        chunk["cross_score"] = float(score)
        reranked.append(chunk)

    # Sort by score and return top 5
    top_chunks = sorted(reranked, key=lambda x: x["cross_score"], reverse=True)[:5]
    return top_chunks
# 🔧 Example Usage
# python
# Copy
# Edit
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # Load model and tokenizer
# model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# # Run reranking
# top_5_chunks = rerank_chunks(query, relevant_chunks, model, tokenizer, device)

# # Print results
# for i, chunk in enumerate(top_5_chunks, 1):
#     print(f"[{i}] Cross Score: {chunk['cross_score']:.4f}")
#     print(f"Text: {chunk['text'][:300]}...\n")



# ==========9.Keyword Extraction =================
def build_keyword_file_from_excel(file_path, output_path='keywords.txt', existing_keywords=None, sheet_name=0):
    if existing_keywords is None:
        existing_keywords = []    
    
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Regex pattern to find 'Keywords: ...'
    keyword_pattern = r"(?i)keywords:\s*([^\n\r]*)"
    
    # Set for uniqueness      
    keyword_set = set(existing_keywords)
                                                                                                                                                                                                                                                                                                                
    for text in df['Full Paper'].dropna():
        match = re.search(keyword_pattern, text)
        if match: 
            extracted = [kw.strip().lower() for kw in match.group(1).split(',') if kw.strip()]
            keyword_set.update(extracted)

    # Save unique sorted keywords to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        for kw in sorted(keyword_set):
            f.write(kw + '\n')

    print(f"[INFO] ✅ Keyword file saved: {output_path}")
    return keyword_set

# ========== 8. MAIN PIPELINE ============
def main():
    print("[INFO] Starting Medical RAG pipeline...")
    
    keywords = load_keywords_from_file("keywords.txt")

    # retrieve_chunks_langchain("Find clinical trial protocols with pmc_id: PMC11358807")


    # pdf_bytes_list = download_pdfs_from_blob(max_pdfs=1)
    # if not pdf_bytes_list:
    #     print("No PDFs found. Exiting.")
    #     return
    
    total_chunks = 0

    # for idx,(filename,pdf_bytes) in enumerate(pdf_bytes_list):
    #     print(f"\n[INFO] Processing Document {idx+1} of {len(pdf_bytes_list)}...")

    #     text = extract_text_from_pdf(pdf_bytes)
    #     if not is_relevant(text, keywords):
    #         print("[INFO] Skipping non-relevant PDF.")
    #         continue

    #     chunks = split_text(text)
    #     print('-' * 200)
    #     print(f"[INFO] Chunked into {len(chunks)} parts.")
        

    #     embeddings = embed_texts(chunks, biobert_tokenizer, biobert_model, device=device)
    #     store_embeddings_pinecone(index,chunks, embeddings, text,filename)

    #     total_chunks += len(chunks)

        
    # print(f"[INFO] ✅ Total relevant chunks stored: {total_chunks}")
    print("\n[INFO] RAG system is ready. Ask medical questions (type 'exit' or 'quit' to stop):")

    session_id = input("\nEnter session ID (or leave blank to use default): ").strip()
    if not session_id:
            session_id = "default"


    while True:
        user_query = input("\nYour question (Type 'exit' or 'quit' to quit): ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
     
        # Step 1: Keyword check
        if not any(kw in user_query.lower() for kw in keywords):
            print("[INFO] ❌ Query is not related to medical context. Please rephrase or ask a different question.")
            continue


        # Step 2: Rewrite query
        preprocessed_query_set =preprocess(user_query)
        preprocessed_query = " ".join(preprocessed_query_set)
        if not preprocessed_query:
            print("[INFO] ❌ I'm a biomedical assistant and cannot process this non-medical query. Please rephrase or ask a different question.")
            continue
        print(f"\n[DEBUG] Preprocessed Query: {preprocessed_query}")

        # 🧠 STEP 1: Retrieve past session context
        raw_context = retrieve_chat_history(index,session_id,user_query,embed_texts,biobert_tokenizer,biobert_model, device=device)
        print("\n[DEBUG] Raw Context History Retrieved:\n", raw_context)

        # 🧠 STEP 2: Parse the session into structured (query, response) tuples
        session_context = parse_session_context(raw_context)
        print("\n[DEBUG] Parsed Session Context:\n", session_context)

        # 🧠 STEP 3: Extract last 3 queries to build query context
        # recent_queries = [q for q, _ in session_context[-3:]]
        # print("\n[DEBUG] Recent User Queries (Last 3):\n", recent_queries)

        # 🧠 STEP 4: Build session-aware query by combining recent queries + current one
        # if not recent_queries:
        contextual_query = user_query
        print("\n[DEBUG] No recent queries found. Using only current user query.")
        # else:
        #     contextual_query = " ".join(recent_queries + [user_query])
        #     print("\n[DEBUG] Final Contextual Query Sent to Embedding & Retrieval:\n", contextual_query)


        print(f"[CACHE] Searching cache for contextual query: {contextual_query}")

        cached = search_cached_response_local(contextual_query,embed_texts,biobert_tokenizer,biobert_model,device)
        if cached:
            print("✅ Cached response found.\n")
            print("[ANSWER]:\n", cached)
            continue  # So it doesn't go to retrieval or generation

        else:
            print("❌ No cached response found. Proceeding with retrieval.")

        # Step 3: Retrieve and rerank
        preprocessed_query_embedding,relevant_chunks = retrieve_chunks(
            contextual_query, index,
        biobert_tokenizer, biobert_model, device=device
        )
        # relevant_chunks = retrieve_chunks_langchain(contextual_query)
         
        
        print("\n🔍 Top Matching Chunks:\n")
        for i, match in enumerate(relevant_chunks, 1):
            print(f"[{i}] Score: {match['score']:.4f}")
            print(f"Text: {match['text'][:300]}...")  # Trim long text for readability
            print(f"Metadata: {match['metadata']}")
            print("-" * 60)


        # reranked_chunks = rerank_chunks(user_query, relevant_chunks, top_k=2)
 
        # print("\n🎯 Reranked Chunks (Cross-Encoder Scores):")
        # for i, chunk in enumerate(reranked_chunks, 1):
        #     print(f"\nReranked Chunk #{i}")
        #     print("-" * 40)
        #     print(f"Text Preview   : {chunk['text'][:200].strip()}...")
        #     print(f"Metadata       : {chunk['metadata']}")
        #     print(f"Cross Score    : {chunk['cross_score']:.4f}")

        reranked_chunks = rerank_chunks(
                contextual_query, 
                relevant_chunks, 
                cross_encoder, 
                biobert_tokenizer, 
                device
            )

        for i, chunk in enumerate(reranked_chunks, 1):
            print(f"[{i}] Cross Score: {chunk['cross_score']:.4f}")
            print(f"Text: {chunk['text'][:300]}...")
            print(f"Metadata: {chunk['metadata']}")
            print("-" * 60)



        # Step 6: Generate answer with session context
        
        answer = generate_answer(reranked_chunks, user_query, session_context=session_context)

        if not is_fallback_answer(answer):
            cache_response_local(contextual_query,answer,embed_texts,biobert_tokenizer,biobert_model,device)
            store_chat_history(index,session_id, user_query, answer,embed_texts,biobert_tokenizer,biobert_model)
            print("\n[ANSWER]:\n", answer)
            print("🆕 Generated, cached, and stored in session history.")

            # async def evaluate_response():
            #     test_data = {
            #         "user_input": user_query,
            #         "response": answer,
            #     } 

            #     metric = AspectCritic(
            #         name="summary_accuracy",
            #         llm=evaluator_llm,
            #         definition="Verify if the summary is accurate."
            #     )

            #     sample = SingleTurnSample(**test_data)
            #     score = await metric.single_turn_ascore(sample)
            #     print("Evaluation Score:", score)

            # Run the async function
            # asyncio.run(evaluate_response())
        else:
            print(f"⚠️ I'm a biomedical assistant and cannot answer '{user_query}' based on the provided context. Please rephrase or ask a different question.")
         
                    
if __name__ == "__main__":     
    # for i in range(100):
    #     print(f"Running iteration {i+1}/100")
        main()
