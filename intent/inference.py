"""Inference script for the offline intent classifier."""
import pickle
import os
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Offline intent classifier for categorize messages.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "model.pkl")
            
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the pickled model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Please run train.py first.")
            return

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("Intent classifier model loaded.")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts intent for the given text.
        Returns a dict with intent and confidence.
        """
        if self.model is None:
            return {"intent": "unknown", "confidence": 0.0, "latency_ms": 0}

        start_time = time.time()
        prediction = self.model.predict([text])[0]
        # Get probabilities
        probs = self.model.predict_proba([text])[0]
        max_prob = max(probs)
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000

        return {
            "intent": str(prediction),
            "confidence": float(max_prob),
            "latency_ms": float(latency_ms)
        }

if __name__ == "__main__":
    # Test inference
    classifier = IntentClassifier()
    test_texts = [
        "Remind me to call the boss",
        "I'm feeling down today",
        "I need to finish this project",
        "How is the weather?",
        "The cat is on the mat"
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"Text: {text} -> Intent: {result['intent']} (Conf: {result['confidence']:.2f}, Latency: {result['latency_ms']:.2f}ms)")
