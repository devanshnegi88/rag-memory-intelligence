"""Training script for the offline intent classifier."""
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(dataset_path: str, model_path: str):
    """
    Trains a TF-IDF + Logistic Regression model.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return

    # Load data
    df = pd.read_csv(dataset_path)
    X = df['text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Train
    logger.info("Training intent classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_model(
        os.path.join(base_dir, "dataset.csv"),
        os.path.join(base_dir, "model.pkl")
    )
