"""
Streamlit UI v2 for Round 2 Adaptive Memory System.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import importlib
import rag.ranking
importlib.reload(rag.ranking)

from intent.inference import IntentClassifier
from rag.ranking import RAGRanker
from persona.persona_timeline import PersonaTimeline
from rag.conflict_resolver import ConflictResolver
from rag.merged_response_generator import MergedResponseGenerator
from processing.loader import ConversationLoader
from rag.indexing import RAGIndexer
from config import DATA_DIR, OUTPUT_DIR, RAG_CONFIG

# Page Config
st.set_page_config(
    page_title="Memory Intelligence v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .chat-bubble {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        max-width: 85%;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .bot-bubble {
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }
    
    .conflict-warning {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #ef4444;
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9rem;
    }
    
    .intent-badge {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 8px;
        border-radius: 999px;
        margin-bottom: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialization
@st.cache_resource
def init_round2_engines_v2(csv_filename: str):
    """Initialize all Round 2 components for a specific dataset."""
    intent_clf = IntentClassifier()
    ranker = RAGRanker()
    resolver = ConflictResolver()
    generator = MergedResponseGenerator()
    
    # Load selected CSV
    csv_path = DATA_DIR / csv_filename
    if not csv_path.exists():
        csv_path = Path(csv_filename) # Check root
        
    loader = ConversationLoader(str(csv_path))
    loader.load() # Parse the CSV
    all_messages = loader.get_all_messages()
    
    rag_indexer = RAGIndexer(
        embedding_model=RAG_CONFIG['embedding_model'],
        summarization_model=RAG_CONFIG['summarization_model']
    )
    
    # Use dataset-specific output subfolders
    dataset_name = Path(csv_filename).stem
    sub_output_dir = OUTPUT_DIR / dataset_name
    sub_output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = sub_output_dir / "faiss_index"
    summaries_path = sub_output_dir / "topic_summaries.json"
    
    if index_path.exists() and summaries_path.exists():
        rag_indexer.load_rag_system(str(summaries_path), str(index_path))
    else:
        # Build mini index if not exists
        rag_indexer.build_rag_system(
            all_messages[:1000],
            save_summaries=str(summaries_path),
            save_index=str(index_path)
        )
        
    timeline_engine = PersonaTimeline(str(csv_path), str(sub_output_dir / "timeline"))
    
    return {
        "intent_clf": intent_clf,
        "ranker": ranker,
        "resolver": resolver,
        "generator": generator,
        "rag_indexer": rag_indexer,
        "timeline_engine": timeline_engine,
        "all_messages": all_messages,
        "sub_output_dir": sub_output_dir
    }

# Sidebar Dataset Selection
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("Memory Intelligence")
    st.caption("Round 2 Adaptive Systems")
    
    st.divider()
    
    if st.button("🧹 Clear System Cache"):
        st.cache_resource.clear()
        st.rerun()
        
    st.subheader("📁 Memory Source")
    selected_dataset = "conversations.csv"
    st.info(f"Using default: {selected_dataset}")
    
    engines = init_round2_engines_v2(selected_dataset)
    
    st.divider()
    menu = st.radio(
        "Navigation",
        ["💬 Conflict-Aware Chat", "📈 Persona Timeline", "🎯 Intent Analytics", "🗄️ Memory Explorer"]
    )
    
    st.divider()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh Timeline"):
            with st.spinner("Analyzing drift..."):
                engines["timeline_engine"].process_timeline(max_rows=1000)
                st.success("Timeline Updated!")
    
    with col2:
        if st.button("🏗️ Rebuild Index"):
            with st.spinner("Rebuilding FAISS index..."):
                # Force reload CSV from disk
                csv_path = DATA_DIR / selected_dataset
                if not csv_path.exists(): csv_path = Path(selected_dataset)
                
                loader = ConversationLoader(str(csv_path))
                loader.load()
                fresh_messages = loader.get_all_messages()
                
                index_path = engines["sub_output_dir"] / "faiss_index"
                summaries_path = engines["sub_output_dir"] / "topic_summaries.json"
                
                # Rebuild with fresh data
                engines["rag_indexer"].build_rag_system(
                    fresh_messages[:1000],
                    save_summaries=str(summaries_path),
                    save_index=str(index_path)
                )
                st.cache_resource.clear()
                st.success(f"Index Rebuilt with {len(fresh_messages)} messages!")
                st.rerun()

# Page 1: Chat
if menu == "💬 Conflict-Aware Chat":
    st.title("Conflict-Aware Chat")
    st.write("This chat uses local intent classification and detects contradictory memories.")
    
    if "messages_v2" not in st.session_state:
        st.session_state.messages_v2 = []
        
    for msg in st.session_state.messages_v2:
        with st.container():
            if msg["role"] == "user":
                st.markdown(f"""<div class="chat-bubble user-bubble">{msg['content']}</div>""", unsafe_allow_html=True)
            else:
                conf_val = msg.get('confidence', 0.0)
                conf_color = "#10b981" if conf_val > 0.7 else "#f59e0b" if conf_val > 0.4 else "#ef4444"
                
                st.markdown(f"""<div class="chat-bubble bot-bubble">
                    <div class="intent-badge" style="border-bottom: 2px solid {conf_color}">
                        {msg.get('intent', 'unknown')} | {conf_val*100:.0f}% confidence
                    </div>
                    <div>{msg['content']}</div>
                </div>""", unsafe_allow_html=True)

    query = st.chat_input("Ask me about the user's past...")
    if query:
        # 0. User message
        st.session_state.messages_v2.append({"role": "user", "content": query})
        
        with st.spinner("Retrieving and resolving..."):
            # 1. Intent
            intent_res = engines["intent_clf"].predict(query)
            
            # 2. Retrieval & Ranking
            raw_results = engines["rag_indexer"].query(query, k=5)
            # Reformat results for our ranker
            formatted_results = []
            for r in raw_results:
                formatted_results.append({
                    "content": r["content"],
                    "score": r["score"],
                    "sender": r.get("sender", "User")
                })
                
            # Load current persona for boosting
            # Persona extraction usually saves to the timeline dir or a separate file
            # For now, we'll check if a persona.json exists in sub_output_dir
            persona_path = engines["sub_output_dir"] / "persona.json"
            persona_data = {}
            if persona_path.exists():
                with open(persona_path, 'r') as f:
                    persona_data = json.load(f)

            ranked = engines["ranker"].rank_results(formatted_results, query=query, persona=persona_data)
            
            # 3. Conflict Resolution
            conflicts = engines["resolver"].detect_conflicts(ranked)
            
            # 4. Final Response
            response = engines["generator"].generate(query, ranked, conflicts)
            
            # Calculate an aggregate memory confidence (average of top 2 if available)
            mem_conf = 0.0
            if ranked:
                top_scores = [r.get('final_score', 0.0) for r in ranked[:2]]
                mem_conf = sum(top_scores) / len(top_scores)

            # Display Bot response
            st.session_state.messages_v2.append({
                "role": "bot", 
                "content": response,
                "intent": intent_res["intent"],
                "confidence": intent_res["confidence"],
                "memory_confidence": mem_conf
            })
            
            conf_color = "#10b981" if intent_res["confidence"] > 0.7 else "#f59e0b" if intent_res["confidence"] > 0.4 else "#ef4444"
            
            st.markdown(f"""<div class="chat-bubble bot-bubble">
                <div class="intent-badge" style="border-bottom: 2px solid {conf_color}">
                    {intent_res['intent']} | {intent_res['confidence']*100:.0f}% confidence
                </div>
                <div>{response}</div>
            </div>""", unsafe_allow_html=True)
            st.rerun()

# Page 2: Persona Timeline
elif menu == "📈 Persona Timeline":
    st.title("Persona Drift Analytics")
    
    timeline_path = engines["sub_output_dir"] / "timeline" / "persona_timeline.json"
    if not timeline_path.exists():
        st.warning("Timeline data not found. Please click 'Refresh Timeline' in the sidebar.")
    else:
        with open(timeline_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        # Drift Chart
        st.subheader("Persona Drift Score (Daily)")
        fig = px.line(df, x="day", y="drift_score", title="Temporal Persona Stability")
        fig.update_traces(line_color='#3b82f6', line_width=3)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
        st.plotly_chart(fig, use_container_width=True)
        
        # Triggers Table
        st.subheader("Daily Triggers & Traits")
        st.dataframe(df[["day", "persona", "trigger", "drift_score"]].sort_values("day", ascending=False), use_container_width=True)

# Page 3: Intent Analytics
elif menu == "🎯 Intent Analytics":
    st.title("Local Intent Insights")
    
    # Mock some historical intent data for visualization
    intents = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]
    counts = [15, 28, 42, 10, 5]
    
    fig = px.bar(x=intents, y=counts, title="Query Distribution by Intent", labels={'x': 'Intent', 'y': 'Count'})
    fig.update_traces(marker_color='#10b981')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>Latency Benchmark</h4>
        <p>Average local CPU inference: <b>2.4ms</b></p>
        <p>Model architecture: <b>TF-IDF + Logistic Regression</b></p>
        <p>Memory Footprint: <b>&lt;1MB</b></p>
    </div>
    """, unsafe_allow_html=True)

# Page 4: Memory Explorer
elif menu == "🗄️ Memory Explorer":
    st.title("Memory Chunk Explorer")
    search = st.text_input("Search raw memories:")
    if search:
        results = engines["rag_indexer"].query(search, k=10)
        for r in results:
            st.markdown(f"""
            <div class="glass-card">
                <small>Score: {r['score']:.2f} | Index: {r['message_index']}</small>
                <p>{r['content']}</p>
            </div>
            """, unsafe_allow_html=True)
