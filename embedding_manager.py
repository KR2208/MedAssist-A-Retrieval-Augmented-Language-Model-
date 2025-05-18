import numpy as np
import faiss
from langchain.embeddings import SentenceTransformerEmbeddings
from tqdm.auto import tqdm
from config import Config

class EmbeddingManager:
    """Manages embeddings and FAISS indices for conditions and drug-condition pairs"""
    
    def __init__(self):
        # Initialize embeddings
        device = Config.setup_cuda()
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": device}
        )
        
        # Initialize indices
        self.cond_index = None
        self.dc_index = None
        self.idx2cond = {}
        self.id2dc = {}
    
    def build_condition_index(self, conditions: list):
        """
        Build FAISS index for medical conditions
        
        Args:
            conditions: List of condition strings
        """
        print(f"Building condition index for {len(conditions)} conditions...")
        
        # Process in batches
        cond_embs = []
        batch_size = Config.BATCH_SIZE_COND
        
        for i in tqdm(range(0, len(conditions), batch_size), desc="Embedding conditions"):
            batch = conditions[i:i+batch_size]
            batch_embs = self.embeddings.embed_documents(batch)
            cond_embs.extend(batch_embs)
        
        # Create FAISS index
        embedding_dimension = len(cond_embs[0])
        self.cond_index = faiss.IndexFlatL2(embedding_dimension)
        self.cond_index.add(np.array(cond_embs, dtype="float32"))
        
        # Create mapping
        self.idx2cond = {i: cond for i, cond in enumerate(conditions)}
        print("Condition index built successfully!")
    
    def build_drug_condition_index(self, review_texts: list, review_meta: list, drug_condition_reviews: dict):
        """
        Build FAISS index for drug-condition pairs
        
        Args:
            review_texts: List of review text chunks
            review_meta: List of review metadata
            drug_condition_reviews: Dictionary mapping (drug, condition) pairs to reviews
        """
        print("Building drug-condition index...")
        
        # Embed all review chunks
        rev_embs = []
        batch_size = Config.BATCH_SIZE_EMBED
        
        for i in tqdm(range(0, len(review_texts), batch_size), desc="Embedding review chunks"):
            batch = review_texts[i:i+batch_size]
            batch_embs = self.embeddings.embed_documents(batch)
            rev_embs.extend(batch_embs)
        
        # Convert to numpy array
        review_embs = np.vstack(rev_embs)
        print(f"Generated embeddings with shape: {review_embs.shape}")
        
        # Get all drug-condition pairs
        dc_pairs = list(drug_condition_reviews.keys())
        print(f"Found {len(dc_pairs)} drug-condition combinations")
        
        # Calculate average embedding for each drug-condition pair
        dc_embs = []
        for drug, cond in tqdm(dc_pairs, desc="Averaging embeddings for drug-condition pairs"):
            # Find all reviews for this drug and condition
            idxs = [i for i, m in enumerate(review_meta)
                    if m["drug"] == drug and m["condition"] == cond]
            
            # Calculate mean embedding
            if idxs:
                dc_embs.append(review_embs[idxs].mean(axis=0))
            else:
                dc_embs.append(np.zeros(review_embs.shape[1], dtype="float32"))
        
        # Stack embeddings and create index
        dc_embs = np.stack(dc_embs)
        self.dc_index = faiss.IndexFlatL2(dc_embs.shape[1])
        self.dc_index.add(dc_embs)
        
        # Create mapping
        self.id2dc = {i: dc_pairs[i] for i in range(len(dc_pairs))}
        print(f"Successfully indexed {len(dc_pairs)} drug-condition combinations")
    
    def detect_condition(self, query: str) -> str:
        """
        Find the most similar condition to the user query
        
        Args:
            query: User query string
            
        Returns:
            Most similar condition name
        """
        if self.cond_index is None:
            raise ValueError("Condition index not built. Call build_condition_index first.")
        
        # Embed the query
        q_vec = self.embeddings.embed_query(query)
        
        # Search for closest condition
        _, idxs = self.cond_index.search(
            np.array([q_vec], dtype="float32"),
            1
        )
        
        return self.idx2cond[idxs[0][0]]
    
    def find_similar_drug_conditions(self, query: str, condition: str, top_k: int = 10) -> list:
        """
        Find similar drug-condition pairs for a given query and condition
        
        Args:
            query: User query
            condition: Detected condition
            top_k: Number of results to return
            
        Returns:
            List of (drug, condition) tuples
        """
        if self.dc_index is None:
            raise ValueError("Drug-condition index not built. Call build_drug_condition_index first.")
        
        # Create query embedding
        q_vec = self.embeddings.embed_query(f"{condition} | {query}")
        
        # Search for similar drug-condition pairs
        _, idxs = self.dc_index.search(
            np.array([q_vec], dtype="float32"),
            top_k * 2  # Get more than needed to filter duplicates
        )
        
        # Return drug-condition pairs
        results = []
        for idx in idxs[0]:
            results.append(self.id2dc[idx])
        
        return results