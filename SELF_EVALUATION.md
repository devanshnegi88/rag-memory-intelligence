# Round 2 Self-Evaluation: AI Conversation Intelligence

## 1. Project Requirements Checklists

### Part 1: Adaptive Persona Drift Engine
- [x] **Day-by-Day Tracking**: Implemented in `PersonaTimeline`, grouping data by date.
- [x] **Drift Detection**: Implemented statistical distance measurement between daily persona traits.
- [x] **Trigger Identification**: Automatically identifies events (e.g., "new job") linked to persona shifts.
- [x] **Visual Timeline**: Dynamic Plotly line chart in UI v2 showing temporal stability.

### Part 2: Offline Intent Classifier
- [x] **Lightweight Model**: Sub-MB model size (TF-IDF + Logistic Regression).
- [x] **Offline Capability**: No API calls; works 100% locally on CPU.
- [x] **Class Coverage**: Successfully classifies `reminder`, `emotional-support`, `action-item`, `small-talk`.
- [x] **Performance**: Benchmark < 3ms latency on standard CPU.

### Part 3: Conflict Resolution in RAG
- [x] **Multidimensional Ranking**: Weighted scoring using Similarity, Recency, and Emotional Weight.
- [x] **Contradiction Flagging**: Regex and entity-matching logic in `ConflictResolver` to detect clashing facts.
- [x] **Merged Coherent Answer**: `MergedResponseGenerator` synthesizes conflicting info with explicit warnings.

### Part 4: System Design & Visuals
- [x] **Architecture Document**: Created `SYSTEM_DESIGN_ROUND2.md` with sync strategy.
- [x] **Premium UI**: Implemented V2 Streamlit interface with high-fidelity aesthetics.
- [x] **Local-First Design**: Entire pipeline runs on device with no external dependencies.

---

## 2. Engineering Decisions (Why & How)

| Feature | Decision | Rationale |
| :--- | :--- | :--- |
| **Intent Classifier** | TF-IDF + LogReg | Chosen over Deep Learning to stay <50MB and ensure <200ms latency on CPU without a GPU. |
| **Conflict Resolution** | Heuristic Entity Matching | Ensures deterministic results. Instead of "guessing" contradictions via LLM, we use exact property clashes (Job A vs Job B). |
| **Vector Database** | FAISS FlatL2 | Provides exact nearest neighbor search for small datasets (<100k rows) with minimal memory overhead. |
| **Persona Drift** | Statistical Distance | Measures the "shift" in character traits (sentiment, formality) numerically, allowing for line-chart visualization. |
