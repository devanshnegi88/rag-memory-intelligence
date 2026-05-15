"""Hybrid retrieval system using FAISS and keyword matching."""
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. Dense embeddings (sentence-transformers + FAISS)
    2. Keyword overlap scoring (TF-IDF)
    
    Weighted scoring combines both approaches.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 dense_weight: float = 0.6, keyword_weight: float = 0.4):
        self.model = SentenceTransformer(model_name)
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        
        self.faiss_index = None
        self.messages = []
        self.message_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def build_index(self, messages: List[Dict], use_tfidf: bool = True) -> None:
        self.messages = messages
        contents = [m['content'] for m in messages if m.get('content')]
        
        if not contents:
            logger.warning("No message content to index.")
            return

        logger.info(f"Building embeddings for {len(contents)} messages...")
        self.message_embeddings = self.model.encode(
            contents, 
            show_progress_bar=False
        ).astype('float32')
        
        logger.info("Building FAISS index...")
        dimension = self.message_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.message_embeddings)
        
        if use_tfidf:
            logger.info("Building TF-IDF matrix...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500, # Increased from 100
                stop_words='english',
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 2) # Capture "software engineer", "San Francisco"
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
    
    def add_messages(self, new_messages: List[Dict]) -> None:
        """Add new messages to the index in real-time."""
        if not new_messages:
            return
            
        start_idx = len(self.messages)
        self.messages.extend(new_messages)
        
        contents = [m['content'] for m in new_messages]
        new_embeddings = self.model.encode(contents).astype('float32')
        
        # Add to FAISS
        if self.faiss_index is not None:
            self.faiss_index.add(new_embeddings)
            
        # Update embeddings array for metadata/save
        if self.message_embeddings is not None:
            import numpy as np
            self.message_embeddings = np.vstack([self.message_embeddings, new_embeddings])
        else:
            self.message_embeddings = new_embeddings
            
        # Note: TF-IDF is not incrementally updated for performance
        # but the semantic (FAISS) search will pick up the new messages immediately.
    
    def retrieve(self, query: str, k: int = 5, 
                use_dense: bool = True, use_keyword: bool = True) -> List[Dict]:
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        results = []
        scores = {}
        
        if use_dense and self.faiss_index is not None:
            dense_results = self._dense_retrieve(query, k * 2)
            for result, score in dense_results:
                msg_idx = result['message_index']
                scores[msg_idx] = scores.get(msg_idx, {})
                scores[msg_idx]['dense'] = score
        
        if use_keyword and self.tfidf_matrix is not None:
            keyword_results = self._keyword_retrieve(query, k * 2)
            for result, score in keyword_results:
                msg_idx = result['message_index']
                scores[msg_idx] = scores.get(msg_idx, {})
                scores[msg_idx]['keyword'] = score
        
        combined_scores = {}
        for msg_idx, score_dict in scores.items():
            dense_score = score_dict.get('dense', 0.0)
            keyword_score = score_dict.get('keyword', 0.0)
            combined = (
                dense_score * self.dense_weight +
                keyword_score * self.keyword_weight
            )
            combined_scores[msg_idx] = {
                'score': combined,
                'dense': dense_score,
                'keyword': keyword_score
            }
        
        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for msg_idx, score_info in sorted_indices[:k]:
            message = self.messages[msg_idx]
            results.append({
                'message_index': msg_idx,
                'content': message['content'],
                'sender': message.get('sender', 'unknown'),
                'score': score_info['score'],
                'dense_score': score_info['dense'],
                'keyword_score': score_info['keyword']
            })
        
        return results
    
    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[Dict, float]]:
        query_embedding = self.model.encode(query).astype('float32')
        
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]), 
            min(k, len(self.messages))
        )
        
        results = []
        max_distance = np.max(distances[0]) if np.max(distances[0]) > 0 else 1.0
        
        # FIXED: iterate over indices[0] instead of indices
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            normalized_score = 1.0 - min(dist / (max_distance + 1e-6), 1.0)
            results.append((self.messages[idx], normalized_score))
        
        return results
    
    def _keyword_retrieve(self, query: str, k: int) -> List[Tuple[Dict, float]]:
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        
        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = scores[idx] / (max_score + 1e-6)
                results.append((self.messages[idx], normalized_score))
        
        return results
    
    def save_index(self, output_dir: str) -> None:
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        
        faiss.write_index(
            self.faiss_index,
            os.path.join(output_dir, 'faiss_index.bin')
        )
        
        metadata = {
            'num_messages': len(self.messages),
            'embedding_dim': self.message_embeddings.shape[1] if self.message_embeddings is not None else 0,
            'messages': [
                {
                    'index': i,
                    'content': m['content'][:200],
                    'sender': m.get('sender', 'unknown')
                }
                for i, m in enumerate(self.messages)
            ]
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_index(self, input_dir: str) -> None:
        import os, json
        self.faiss_index = faiss.read_index(
            os.path.join(input_dir, 'faiss_index.bin')
        )
        
        with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.messages = [
            {
                'content': m['content'],
                'sender': m['sender'],
                'message_index': i
            }
            for i, m in enumerate(metadata['messages'])
        ]
