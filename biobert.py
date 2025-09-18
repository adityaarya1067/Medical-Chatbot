from langchain.embeddings.base import Embeddings
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

class BioBERTLangChainEmbedder(Embeddings):
    def __init__(self, tokenizer, model, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device

    def embed_query(self, text: str) -> List[float]:
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512,  padding=True).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)
        return cls_embedding.squeeze().cpu().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = self.embed_query(text)
            embeddings.append(emb)
        return embeddings
