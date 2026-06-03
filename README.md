# Memory Intelligence & Conflict Resolution

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
- **Local-First**: All models are <50MB and run locally on CPU, ensuring zero external API latency and maximum privacy.- Tracks: frequency, exact message references

#### Personal Facts
- Explicit mentions only (no inference)
- Categories: name, age, location, occupation, interests, etc.
- Pattern matching for "I am...", "I live in...", "I work as..."

#### Personality Traits
- **Positive sentiment**: Happy, excited, enthusiastic
- **Negative sentiment**: Frustrated, sad, angry
- **Curiosity**: Question frequency
- **Enthusiasm**: Exclamation mark usage
- **Politeness**: Thank you, please, sorry patterns

#### Communication Style
- **Verbosity**: Concise (< 3 words avg), normal, verbose (> 20 words avg)
- **Formality**: Formal vs casual (based on exclamation marks, caps)
- **Emoji usage**: Boolean flag
- **Message length statistics**: Average and variance

**Persona output**:
```json
{
  "habits": [
    {
      "trait": "I usually drink coffee in the morning",
      "evidence": [5, 23, 45],
      "frequency": 3
    }
  ],
  "personal_facts": [
    {
      "fact": "Engineer",
      "category": "occupation",
      "evidence": [12, 34]
    }
  ],
  "personality_traits": [
    {
      "trait": "curious inquisitive",
      "evidence": [2, 8, 15, 22],
      "strength": 0.45
    }
  ],
  "communication_style": {
    "avg_message_length": 45,
    "verbosity": "normal",
    "formality": "casual",
    "emoji_usage": true
  }
}
```

### 💬 Part 3: Chatbot API

#### Intent Classification
- **persona**: "Tell me about yourself" → returns extracted persona
- **habits**: "What are your habits?" → returns habit patterns
- **general**: "What did we discuss?" → uses RAG retrieval

#### Endpoints

**POST /chat**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my habits?"}'

# Response
{
  "query": "What are my habits?",
  "intent": "habits",
  "response": "**Identified Habits:**\n- I usually drink coffee in the morning (mentioned 3 times)\n- I often work late (mentioned 2 times)",
  "sources": [
    {
      "type": "persona",
      "component": "habits",
      "habit_count": 8
    }
  ],
  "confidence": 0.8
}
```

**GET /topics**
```bash
curl http://localhost:5000/topics

# Response
{
  "total": 3,
  "topics": [
    {
      "topic_id": 0,
      "label": "Work & Projects",
      "start_idx": 0,
      "end_idx": 25,
      "message_count": 26
    }
  ]
}
```

**GET /persona**
```bash
curl http://localhost:5000/persona
```

**GET /health**
```bash
curl http://localhost:5000/health
# Response: {"status": "ok"}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input CSV                              │
│         (One conversation per row)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Processing Module                           │
│           (Load & Parse Messages)                        │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐   ┌──────────────────────┐
│  RAG System       │   │ Persona Extraction   │
├───────────────────┤   ├──────────────────────┤
│1. Topic Detection │   │1. Extract Habits     │
│2. Summarization   │   │2. Extract Facts      │
│3. Build Index     │   │3. Extract Traits     │
│4. Hybrid Retrieval│   │4. Communication Style│
└────────┬──────────┘   └──────────┬───────────┘
         │                         │
         │    ┌────────────────────┘
         │    │
         ▼    ▼
      ┌───────────────────┐
      │  Chatbot API      │
      ├───────────────────┤
      │ Intent Detection  │
      │ Query Answering   │
      │ Response Gen      │
      └───────────────────┘
```

### Module Breakdown

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `processing/` | Data loading & parsing | `ConversationLoader` |
| `rag/` | Topic detection & retrieval | `TopicDetector`, `SummarizationEngine`, `HybridRetriever`, `RAGIndexer` |
| `persona/` | Persona extraction | `PersonaExtractor` |
| `chatbot/` | Flask API | `ChatbotAPI` |

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone and navigate**:
```bash
cd "rag system"
```

2. **Create virtual environment** (recommended):
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag-system python=3.9
conda activate rag-system
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note on GPU support**: By default, FAISS uses CPU. For GPU support, replace `faiss-cpu` with `faiss-gpu` in requirements.txt.

---

## Usage

### Quick Start with Streamlit UI 🚀

The easiest way to interact with the chatbot is using the Streamlit interface:

```bash
# Windows
run_streamlit.bat

# Mac/Linux or cross-platform
python run_streamlit.py

# Or manually
streamlit run streamlit_app.py
```

The UI opens at: **http://localhost:8501**

**Features:**
- 💬 Interactive chat with intent classification
- 👤 Persona explorer with traits and habits
- 📚 Topics visualization and analysis
- 📊 System dashboard with metrics

See [STREAMLIT_QUICKSTART.md](STREAMLIT_QUICKSTART.md) for details.

---

### Basic Usage

1. **Prepare CSV file**

Expected format (example):
```csv
date,conversation
2024-01-01,"[{""sender"": ""user"", ""content"": ""Hi there!""}, {""sender"": ""assistant"", ""content"": ""Hello!""}]"
```

Or simpler format:
```csv
date,conversation
2024-01-01,"user: Hi there!
assistant: Hello! How are you?"
```

2. **Run the system**:

```bash
# Full pipeline: Process data + Launch API
python main.py --csv data/conversations.csv --port 5000

# Only process data (no API)
python main.py --csv data/conversations.csv --no-api

# With custom output directory
python main.py --csv data/conversations.csv --output results/ --port 8000
```

3. **Check outputs** in `outputs/`:
   - `topic_summaries.json` - Topic detection results
   - `topic_summaries_checkpoints.json` - 100-message checkpoint summaries
   - `persona.json` - Extracted persona
   - `faiss_index/` - FAISS index and metadata

4. **Query the API**:

```bash
# In another terminal
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about yourself"}'
```

---

## API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Chat (POST /chat)

Submit a query and get an intelligent response.

**Request**:
```json
{
  "query": "What are your habits?"
}
```

**Response**:
```json
{
  "query": "What are your habits?",
  "intent": "habits",
  "response": "**Identified Habits:**\n- I usually drink coffee...",
  "sources": [
    {
      "type": "persona",
      "component": "habits",
      "habit_count": 8
    }
  ],
  "confidence": 0.8
}
```

**Intent Types**:
- `persona`: Answers about personality, traits, facts
- `habits`: Answers about behavioral patterns
- `general`: Retrieves from conversation context

---

#### 2. Get Topics (GET /topics)

List all detected topics.

**Response**:
```json
{
  "total": 5,
  "topics": [
    {
      "topic_id": 0,
      "label": "Work Projects",
      "start_idx": 0,
      "end_idx": 23,
      "message_count": 24
    }
  ]
}
```

---

#### 3. Get Persona (GET /persona)

Retrieve extracted persona.

**Response**:
```json
{
  "habits": [...],
  "personal_facts": [...],
  "personality_traits": [...],
  "communication_style": {...}
}
```

---

#### 4. Health Check (GET /health)

Simple health probe.

**Response**:
```json
{
  "status": "ok"
}
```

---

## Project Structure

```
rag system/
├── processing/              # Data loading
│   ├── __init__.py
│   └── loader.py           # ConversationLoader
├── rag/                    # RAG system
│   ├── __init__.py
│   ├── topic_detection.py  # TopicDetector
│   ├── summarization.py    # SummarizationEngine
│   ├── retrieval.py        # HybridRetriever
│   └── indexing.py         # RAGIndexer (coordinator)
├── persona/                # Persona extraction
│   ├── __init__.py
│   └── extractor.py        # PersonaExtractor
├── chatbot/                # Flask API
│   ├── __init__.py
│   └── api.py              # ChatbotAPI
├── outputs/                # Generated outputs
│   ├── topic_summaries.json
│   ├── topic_summaries_checkpoints.json
│   ├── persona.json
│   └── faiss_index/
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Technical Details

### Topic Detection Algorithm

**Problem**: Conversations often jump between topics. We need to detect where topic shifts occur without treating the entire conversation as one topic.

**Solution**: Three-factor detection model:

1. **Semantic Similarity** (60% weight)
   - Slides a window of 5 messages
   - Compares average embedding before and after current position
   - If cosine similarity drops, likely topic shift
   - Uses sentence-transformers (all-MiniLM-L6-v2)

2. **Keyword Shift** (30% weight)
   - Extracts keywords before/after window
   - Computes Jaccard distance (1 - overlap)
   - High distance = topic shift

3. **Conversational Markers** (10% bonus)
   - Detects phrases: "btw", "anyway", "speaking of", etc.
   - Boosts shift score if found

**Example**:
```
Messages 0-4:   "I love hiking... weekends are great..."
                 → Keywords: [hiking, weekends, nature]
                 → Avg embedding: [0.12, -0.43, ...]

                 SHIFT DETECTED (similarity = 0.45)

Messages 5-9:   "By the way, work has been stressful..."
                 → Keywords: [work, stress, meeting]
                 → Avg embedding: [0.61, -0.12, ...]
```

### Retrieval System

**Hybrid Retrieval**:
1. **Dense Retrieval**: Query embedding → FAISS search → top-k candidates
   - L2 distance normalized to 0-1
   - Better for semantic similarity

2. **Keyword Retrieval**: Query TF-IDF vector → cosine similarity → top-k candidates
   - Better for exact term matching

3. **Score Combination**: 
```
final_score = 0.6 * dense_score + 0.4 * keyword_score
```

Results are ranked by combined score.

### Persona Extraction

**Core Principle**: Evidence-based extraction only.

**Habit Detection**:
```python
# Looks for: "I always/usually/often [behavior]"
# Requires: 2+ mentions (configurable)
# Output: trait + message indices
```

**Personal Fact Extraction**:
```python
# Only explicit mentions using regex patterns
# "I am [fact]" → extracts [fact]
# "I live in [location]" → extracts location
# Never infers or generalizes
```

**Trait Detection**:
```python
# Positive sentiment: happiness, excitement indicators
# Negative sentiment: frustration, sadness indicators
# Curiosity: question marks
# Enthusiasm: exclamation marks
# Politeness: "please", "thank you", etc.
```

**Communication Style**:
```python
# avg_message_length: mean of all message lengths
# verbosity: categorized by avg word count
# formality: based on punctuation patterns
# emoji_usage: boolean flag
```

---

## Deployment

### Local Deployment

```bash
python main.py --csv data.csv --host 0.0.0.0 --port 5000
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--csv", "/data/conversations.csv", "--host", "0.0.0.0", "--port", "5000"]
```

Build and run:
```bash
docker build -t rag-system .
docker run -p 5000:5000 -v /path/to/data:/data rag-system
```

### Render Deployment

1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: rag-system
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python main.py --csv data/conversations.csv --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: "3.9"
```

2. **Push to GitHub**

3. **Connect on Render**:
   - New → Web Service
   - Connect GitHub repo
   - Select repo and deploy

4. **Set environment**:
   - Upload CSV file to Render storage
   - Update CSV path in start command

---

## Advanced Usage

### Custom Models

```python
from rag.indexing import RAGIndexer

indexer = RAGIndexer(
    embedding_model='sentence-transformers/all-mpnet-base-v2',  # Larger model
    summarization_model='google/pegasus-arxiv',  # Different summarizer
    window_size=7,  # Larger window
    checkpoint_size=200  # Larger checkpoints
)

result = indexer.build_rag_system(messages)
```

### Fine-tuning Topic Detection

```python
from rag.topic_detection import TopicDetector

detector = TopicDetector(
    similarity_threshold=0.5,  # Lower threshold = more topics
    keyword_weight=0.5  # Higher weight for keyword shifts
)

topics = detector.detect_topics(messages)
```

### Custom Persona Rules

```python
from persona.extractor import PersonaExtractor

extractor = PersonaExtractor(min_repetitions=3)  # Require 3+ repetitions

persona = extractor.extract(messages)
```

---

## Troubleshooting

### FAISS Index Issues
```
ImportError: cannot import name 'IndexFlatL2'
```
**Solution**: Reinstall FAISS
```bash
pip uninstall faiss-cpu
pip install faiss-cpu
```

### Memory Issues with Large Conversations
```python
# Process in batches
batch_size = 1000
for i in range(0, len(messages), batch_size):
    batch = messages[i:i+batch_size]
    rag_indexer.build_rag_system(batch)
```

### Slow Summarization
- Use smaller checkpoint sizes
- Use lighter model: `distilbart-cnn-6-6` instead of `bart-large-cnn`

---

## Performance Metrics

Typical performance on standard hardware:

| Operation | Time | Memory |
|-----------|------|--------|
| Load 1000 messages | < 1s | 10 MB |
| Topic detection | 5-10s | 50 MB |
| Summarization | 1-2 min | 200 MB |
| FAISS indexing | 3-5s | 100 MB |
| Query retrieval | < 100ms | 5 MB |

---

## Contributing

Contributions welcome! Areas for enhancement:

1. **Better topic detection**: Try temporal signals or semantic graphs
2. **Improved summarization**: Fine-tuned models for conversations
3. **Persona enrichment**: Social network analysis, tone detection
4. **API improvements**: WebSocket support, batch queries, authentication

---

## License

MIT License - feel free to use and modify.

---

## Contact

For questions or issues, please create a GitHub issue or email the maintainer.

---

## Citation

If you use this system in research, please cite:

```bibtex
@software{rag_conversation_system_2024,
  title = {RAG-based Conversation Intelligence System with Persona Extraction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rag-system}
}
```
#   r a g - s y s t e m 
 
 #   r a g - m e m o r y - i n t e l l i g e n c e 
 
 
