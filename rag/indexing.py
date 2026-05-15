"""RAG indexing coordinator."""
import json
import logging
from typing import List, Dict
from .topic_detection import TopicDetector
from .summarization import SummarizationEngine
from .retrieval import HybridRetriever
import os

logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    Coordinates topic detection, summarization, and retrieval indexing.
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2',
                 summarization_model: str = 'facebook/bart-large-cnn',
                 window_size: int = 5,
                 checkpoint_size: int = 100,
                 gemini_api_key: str = None):
        """
        Initialize RAG indexer.
        
        Args:
            embedding_model: Sentence transformer model
            summarization_model: Summarization model
            window_size: Sliding window size for topic detection
            checkpoint_size: Messages per checkpoint for summaries
            gemini_api_key: Optional Gemini API key
        """
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.window_size = window_size
        self.checkpoint_size = checkpoint_size
        
        self.topic_detector = TopicDetector(
            model_name=embedding_model,
            window_size=window_size
        )
        self.summarizer = SummarizationEngine(
            model_name=summarization_model,
            gemini_api_key=gemini_api_key
        )
        self.retriever = HybridRetriever(
            model_name=embedding_model
        )
        
        self.topics = []
        self.checkpoints = []
        self.messages = []
    
    def build_rag_system(self, messages: List[Dict], 
                        save_summaries: str = None,
                        save_index: str = None) -> Dict:
        """
        Build complete RAG system.
        
        Args:
            messages: List of message dicts
            save_summaries: Path to save summaries JSON
            save_index: Path to save FAISS index
            
        Returns:
            Dict with keys: topics, checkpoints, retriever
        """
        self.messages = messages
        
        logger.info(f"Starting RAG indexing for {len(messages)} messages...")
        
        # Step 1: Detect topics
        logger.info("Step 1: Detecting topics...")
        self.topics = self.topic_detector.detect_topics(messages)
        logger.info(f"  Found {len(self.topics)} topics")
        
        # Step 2: Summarize topics
        logger.info("Step 2: Generating topic summaries...")
        self.topics = self.summarizer.summarize_topics(self.topics)
        logger.info(f"  Generated {len(self.topics)} summaries")
        
        # Step 3: Create checkpoint summaries
        logger.info(f"Step 3: Creating {self.checkpoint_size}-message checkpoints...")
        self.checkpoints = self.summarizer.create_checkpoint_summaries(
            messages,
            checkpoint_size=self.checkpoint_size
        )
        logger.info(f"  Created {len(self.checkpoints)} checkpoints")
        
        # Step 4: Build retrieval index
        logger.info("Step 4: Building retrieval index...")
        self.retriever.build_index(messages, use_tfidf=True)
        logger.info("  Index built successfully")
        
        # Step 5: Save outputs
        if save_summaries:
            logger.info(f"Saving summaries to {save_summaries}...")
            self.summarizer.save_summaries(self.topics, save_summaries)
            self._save_checkpoints(self.checkpoints, 
                                  save_summaries.replace('.json', '_checkpoints.json'))
        
        if save_index:
            logger.info(f"Saving index to {save_index}...")
            self.retriever.save_index(save_index)
        
        logger.info("RAG system built successfully!")
        
        return {
            'topics': self.topics,
            'checkpoints': self.checkpoints,
            'retriever': self.retriever,
            'message_count': len(messages),
            'topic_count': len(self.topics),
            'checkpoint_count': len(self.checkpoints)
        }

    def load_rag_system(self, summaries_path: str, index_path: str) -> None:
        """
        Load RAG system from saved state.
        """
        logger.info(f"Loading RAG system from {summaries_path} and {index_path}...")
        
        # Load summaries
        with open(summaries_path, 'r') as f:
            self.topics = json.load(f)
            
        # Load checkpoints
        checkpoints_path = summaries_path.replace('.json', '_checkpoints.json')
        if os.path.exists(checkpoints_path):
            with open(checkpoints_path, 'r') as f:
                self.checkpoints = json.load(f).get('checkpoints', [])
        
        # Load index
        self.retriever.load_index(index_path)
        self.messages = self.retriever.messages
        
        logger.info("RAG system loaded successfully!")
    
    def _save_checkpoints(self, checkpoints: List[Dict], output_path: str) -> None:
        """Save checkpoint summaries to JSON."""
        output_data = {
            'total_checkpoints': len(checkpoints),
            'checkpoint_size': self.checkpoint_size,
            'checkpoints': checkpoints
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def query(self, query: str, k: int = 5) -> List[Dict]:
        """
        Query the RAG system.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of retrieved messages
        """
        return self.retriever.retrieve(query, k=k)
    
    def get_topic_summary(self, topic_id: int) -> Dict:
        """
        Get summary for a specific topic.
        
        Args:
            topic_id: Topic ID
            
        Returns:
            Topic dict with summary
        """
        for topic in self.topics:
            if topic['topic_id'] == topic_id:
                return topic
        
        return None
    
    def get_checkpoint_summary(self, checkpoint_id: int) -> Dict:
        """
        Get summary for a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint dict with summary
        """
        for checkpoint in self.checkpoints:
            if checkpoint['checkpoint_id'] == checkpoint_id:
                return checkpoint
        
        return None
