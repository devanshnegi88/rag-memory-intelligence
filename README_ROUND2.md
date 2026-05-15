# Round 2: Memory Intelligence & Conflict Resolution

This extension upgrades the RAG platform from simple retrieval to **Conversation Intelligence**. It implements adaptive personality tracking, offline intent classification, and a sophisticated conflict-resolution pipeline.

## 🧠 Intelligence Modules

### 1. Adaptive Persona Drift Engine
Located in `/persona`, this module tracks the evolution of the user's character over time.
- **Temporal Grouping**: Analyzes conversations day-by-day.
- **Drift Scoring**: Uses statistical distance between character traits (sentiment, formality, verbosity) to detect shifts.
- **Trigger Detection**: Links persona changes to specific events (e.g., "new job", "relocation").

### 2. Offline Intent Classifier
Located in `/intent`, this is a privacy-first, local inference engine.
- **Architecture**: TF-IDF vectorization with a Logistic Regression classifier.
- **Performance**: < 3ms latency on local CPU.
- **Categories**: `reminder`, `emotional-support`, `action-item`, `small-talk`.

### 3. Conflict-Aware RAG Resolver
Located in `/rag`, this solves the "hard retrieval problem" of contradictory memories.
- **Weighted Ranking**: Prioritizes chunks based on **Recency** and **Emotional Weight**.
- **Conflict Detection**: Heuristic entity matching identifies clashing facts (e.g., "Job: Pilot" vs "Job: Teacher").
- **Merged Synthesis**: Generates a coherent response that explicitly highlights memory inconsistencies.

## 🚀 How to Run (Round 2)

### 1. Prerequisites
Ensure you have the Round 2 dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Intelligence Dashboard
Run the premium V2 interface:
```bash
python -m streamlit run streamlit_app_v2.py
```

### 3. Testing the Intelligence
- **Persona**: Use the **"📈 Persona Timeline"** tab to see drift analytics.
- **Intent**: Type a message in the chat and look for the **Intent Badge**.
- **Conflict**: Add two contradictory facts to your CSV, click **"🏗️ Rebuild Index"**, and ask about that topic.

## 🏗️ Architecture Decisions
- **Deterministic over "Magic"**: We use entity matching and statistical distance instead of unpredictable LLM calls for conflict resolution and drift.
- **Local-First**: All models are <50MB and run locally on CPU, ensuring zero external API latency and maximum privacy.
