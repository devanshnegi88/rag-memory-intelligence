"""Ranking module for RAG results."""
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RAGRanker:
    """
    Ranks retrieved chunks using similarity, recency, and emotional weight.
    """
    def rank_results(self, results: List[Dict], query: str = "", persona: Dict = None) -> List[Dict]:
        """
        results: List of dicts with 'content', 'score' (similarity), 'date' (optional).
        query: Original user query for keyword matching.
        persona: Optional persona dict to boost relevant facts.
        """
        ranked_results = []
        
        for res in results:
            content = res['content']
            # 1. Similarity score
            sim_score = res.get('score', 0.5)
            
            # 2. Recency score (newer messages get higher score)
            # Find the max index among results to normalize
            max_idx = max([r.get('message_index', 1000) for r in results]) if results else 1000
            current_idx = res.get('message_index', 0)
            recency_score = current_idx / max_idx if max_idx > 0 else 0.5
            
            # 3. Emotional weight
            emotional_weight = self._calculate_emotional_weight(content)
            
            # 4. Persona Boost (Round 2 specific)
            persona_boost = 0.0
            if persona and 'personal_facts' in persona:
                for fact in persona['personal_facts']:
                    if fact['fact'].lower() in content.lower():
                        persona_boost += 0.3
            
            # 5. Exact Query Keyword Boost (New)
            # If the user asks "job" and the chunk has "job" or an occupation, boost it.
            keyword_boost = 0.0
            query_terms = query.lower().split()
            for term in query_terms:
                if len(term) >= 3 and term in content.lower():
                    keyword_boost += 0.2
            
            # Combined score: 30% Sim, 15% Recency, 10% Emotion, 25% Persona, 20% Keyword
            combined_score = (sim_score * 0.3) + (recency_score * 0.15) + (emotional_weight * 0.1) + \
                             (min(persona_boost, 1.0) * 0.25) + (min(keyword_boost, 1.0) * 0.2)
            
            res['final_score'] = combined_score
            ranked_results.append(res)
            
        return sorted(ranked_results, key=lambda x: x['final_score'], reverse=True)

    def _calculate_emotional_weight(self, text: str) -> float:
        """
        Simple heuristic for emotional weight: exclamation marks, caps, emotional keywords.
        """
        weight = 0.0
        if '!' in text: weight += 0.2
        if text.isupper(): weight += 0.3
        
        emotional_keywords = ['love', 'hate', 'happy', 'sad', 'angry', 'scared', 'excited', 'frustrated']
        for word in emotional_keywords:
            if word in text.lower():
                weight += 0.1
                
        return min(weight, 1.0)
