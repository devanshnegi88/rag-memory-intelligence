import pandas as pd
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def label_message(text):
    text = text.lower()
    
    # 1. Reminder
    if any(k in text for k in ["remind", "forget", "reminder", "don't let me"]):
        return "reminder"
    
    # 2. Emotional Support
    if any(k in text for k in ["feel", "sad", "happy", "stress", "anxious", "depressed", "lonely", "excited"]):
        return "emotional-support"
    
    # 3. Action Item
    if any(k in text for k in ["task", "todo", "to do", "finish", "complete", "need to", "should start", "must", "have to", "plan to", "going to"]):
        return "action-item"
    
    # 4. Small Talk (Questions and greetings)
    if any(k in text for k in ["hi", "hello", "how are", "weather", "tell me", "did i mention", "have we talked", "what is", "who is", "thanks"]):
        return "small-talk"
    
    return "unknown"

def process_and_train(csv_path, output_model_path):
    logger.info(f"Loading conversations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    all_user_messages = []
    
    # Extract messages from the 'conversation' column
    for conv in df['conversation'].dropna().head(10000): # Process more rows for better coverage
        lines = str(conv).split('\n')
        for line in lines:
            if "User 1:" in line:
                msg = line.split("User 1:")[1].strip()
                if len(msg.split()) > 2:
                    all_user_messages.append(msg)
    
    logger.info(f"Extracted {len(all_user_messages)} messages. Labeling...")
    
    labeled_data = []
    for msg in all_user_messages:
        label = label_message(msg)
        labeled_data.append({"text": msg, "label": label})
    
    labeled_df = pd.DataFrame(labeled_data)
    
    # CAP UNKNOWN SAMPLES to prevent class imbalance from drowning out specifics
    unknowns = labeled_df[labeled_df['label'] == 'unknown']
    others = labeled_df[labeled_df['label'] != 'unknown']
    
    if len(unknowns) > 5000:
        unknowns = unknowns.sample(5000, random_state=42)
    
    labeled_df = pd.concat([others, unknowns])
    
    counts = labeled_df['label'].value_counts()
    logger.info(f"Balanced Label Distribution:\n{counts}")
    
    # Training
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    logger.info("Training on labeled conversation data...")
    pipeline.fit(labeled_df['text'], labeled_df['label'])
    
    with open(output_model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    logger.info(f"Model successfully trained and saved to {output_model_path}")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root_dir, "data", "conversations.csv")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    
    process_and_train(csv_path, model_path)
