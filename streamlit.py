import streamlit as st
import torch
from main import (
    preprocess, search_cached_response_local, retrieve_chunks,
    generate_answer, cache_response_local, is_fallback_answer,
    retrieve_chat_history, store_chat_history,
    biobert_model, biobert_tokenizer,parse_session_context,index,embed_texts,device
)

from sentence_transformers import CrossEncoder
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_keywords_from_file(path='keywords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set(kw.strip().lower() for kw in f if kw.strip())


keywords = load_keywords_from_file("keywords.txt")

st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🧠 Medical Chatbot (BioBERT + Pinecone + RAG)")

# Session ID
session_id = st.text_input("Enter your session ID (or create a new one):", value="default_session").strip()

# Chat input
user_query = st.text_input("Ask your medical question here:").strip()

if user_query:
    st.markdown(f"**Query:** {user_query}")

    # Step 1: Keyword Check
    if not any(kw in user_query.lower() for kw in keywords):
        st.warning("❌ Query is not related to medical context. Please rephrase.")
    else:
        # Step 2: Rewrite query
        preprocessed_query_set = preprocess(user_query)
        preprocessed_query = " ".join(preprocessed_query_set).strip()

        if not preprocessed_query:
            st.error("❌ Query could not be processed. Possibly non-medical.")
        else:
            st.markdown(f"`Preprocessed:` {preprocessed_query}")

            # Step 3: Retrieve session context
            start = time.time()
            raw_context = retrieve_chat_history(index,session_id,user_query,embed_texts,biobert_tokenizer,biobert_model, device=device)
            end = time.time()
            print(f"✅ Session context retrieved in {end - start:.2f} seconds.")
            st.markdown("### 📄 Retrieved Context Chunks")
            st.markdown(f"```text\n{raw_context}\n```")

            session_context = parse_session_context(raw_context)
            recent_queries = [q for q, _ in session_context[-3:]]
            contextual_query = " ".join(recent_queries + [user_query])

            # Step 4: Check cache
            start=time.time()
            cached = search_cached_response_local(contextual_query,embed_texts,biobert_tokenizer,biobert_model,device)
            if cached:
                end = time.time()
                print(f"✅ Cached response found in {end - start:.2f} seconds.")
                st.success("✅ Cached response found.")
                st.markdown(f"### 🤖 Answer:\n{cached.strip()}")
            else:
                st.info("🔎 No cached response. Proceeding with retrieval...")

                # Step 5: Retrieve and rerank
                start = time.time()
                st.spinner("🔄 Retrieving relevant chunks...")
                embedding, relevant_chunks = retrieve_chunks(
                    contextual_query, index,biobert_tokenizer, biobert_model, device=device
                )
                end = time.time()
                print(f"✅ Relevant chunks retrieved in {end - start:.2f} seconds.")

                if not relevant_chunks:
                    st.warning("⚠️ No relevant chunks found.")
                # else:
                #     pairs = [(user_query, chunk) for chunk, _ in relevant_chunks]
                #     scores = reranker.predict(pairs)

                #     min_score = min(scores)
                #     max_score = max(scores)
                #     normalized_scores = [(s - min_score) / (max_score - min_score + 1e-9) for s in scores]

                #     scored_chunks = sorted(zip(relevant_chunks, normalized_scores), key=lambda x: x[1], reverse=True)

                    # with st.expander("🔍 Source Chunks Used"):
                    #     for i, ((chunk_text, metadata), score) in enumerate(scored_chunks[:5]):
                    #         if round(score, 3) >= 0.75:
                    #             st.markdown(f"**Chunk {i+1} | Confidence Score: {round(score, 3)}**")
                    #             st.markdown(f"- DOI: {metadata.get('doi', '')}")
                    #             st.markdown(f"- PMC: {metadata.get('pmc_id', '')}")
                    #             st.text(chunk_text[:300] + "...\n")

                    # Step 6: Generate Answer with Context Awareness
                    start = time.time()
                    with st.spinner("🔄 Generating answer from relevant chunks and session context..."):
                        answer = generate_answer(relevant_chunks, user_query, session_context=session_context).strip()
                    end = time.time()
                    print(f"✅ Answer generated in {end - start:.2f} seconds.")
                    

                    if not is_fallback_answer(answer):
                        st.success("✅ Answer generated successfully!")
                        st.markdown(f"### 🤖 Answer:\n{answer}")
                        cache_response_local(contextual_query,answer,embed_texts,biobert_tokenizer,biobert_model,device)
                        store_chat_history(index,session_id, user_query, answer,embed_texts,biobert_tokenizer,biobert_model)
                        st.success("🆕 Generated, cached, and session history updated.")
                    else:
                        st.error(f"⚠️ I'm a biomedical assistant and cannot answer '{user_query}' based on the provided context. Please rephrase or ask a different question.")

