"""Demo script for Round 2 features."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from persona.persona_timeline import PersonaTimeline
from intent.inference import IntentClassifier
from rag.ranking import RAGRanker
from rag.conflict_resolver import ConflictResolver
from rag.merged_response_generator import MergedResponseGenerator

def run_demo():
    print("=== Round 2 Feature Demo ===")
    
    # 1. Intent Classifier
    print("\n[1] Testing Offline Intent Classifier...")
    classifier = IntentClassifier()
    query = "Remind me to buy groceries"
    result = classifier.predict(query)
    print(f"Query: '{query}' -> Detected Intent: {result['intent']} (Latency: {result['latency_ms']:.2f}ms)")
    
    # 2. Persona Timeline
    print("\n[2] Running Adaptive Persona Timeline Analysis...")
    timeline_engine = PersonaTimeline(
        conversations_path="data/conversations.csv",
        output_dir="outputs/timeline"
    )
    # Process just the first few days for the demo
    print("Processing first 500 rows for demo...")
    timeline = timeline_engine.process_timeline(max_rows=500)
    print(f"Generated timeline for {len(timeline)} days.")
    print("Latest Persona State Sample:")
    if timeline:
        last_day = timeline[-1]
        print(f"  Day: {last_day['day']}")
        print(f"  Persona: {last_day['persona']}")
        print(f"  Trigger: {last_day['trigger']}")
        print(f"  Drift Score: {last_day.get('drift_score', 'N/A')}")

    # 3. Conflict-Aware RAG
    print("\n[3] Testing Conflict-Aware RAG Resolver...")
    # Mocking retrieved results for demonstration of the logic
    mock_results = [
        {"content": "The user lives in Delhi", "score": 0.9, "sender": "User 1"},
        {"content": "The user moved to Mumbai last week", "score": 0.85, "sender": "User 1"}
    ]
    
    ranker = RAGRanker()
    resolver = ConflictResolver()
    generator = MergedResponseGenerator()
    
    ranked = ranker.rank_results(mock_results)
    conflicts = resolver.detect_conflicts(ranked)
    response = generator.generate("Where does the user live?", ranked, conflicts)
    
    print(f"Query: 'Where does the user live?'")
    print(f"Response: {response}")
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()
