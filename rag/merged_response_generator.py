"""Merged response generator for RAG results."""
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MergedResponseGenerator:
    """
    Generates a coherent response by merging retrieved chunks and resolving conflicts.
    """
    def generate(self, query: str, results: List[Dict], conflicts: List[Tuple[Dict, Dict, str]]) -> str:
        """
        Combines information and explicitly mentions conflicts.
        """
        # Filter out very short messages or generic greetings from results
        filtered_results = [
            r for r in results 
            if len(r['content'].split()) > 3 and not any(g in r['content'].lower() for g in ["hi!", "how are you?", "hello"])
        ]
        
        # Fallback to unfiltered if all were noise
        display_results = filtered_results if filtered_results else results
        
        if not display_results:
            return "I couldn't find any specific information about that."

        response_parts = []
        
        # 1. Main information from top results
        top_results = display_results[:2]
        info_texts = [f"'{r['content']}'" for r in top_results]
        
        if len(info_texts) > 1:
            response_parts.append(f"Based on our conversations, you mentioned {info_texts[0]} and {info_texts[1]}.")
        else:
            response_parts.append(f"You previously mentioned: {info_texts[0]}.")
        
        # 2. Address conflicts
        if conflicts:
            conflict_msg = " However, I noticed some inconsistencies:"
            for msg1, msg2, reason in conflicts:
                conflict_msg += f"\n- {reason} (previously: '{msg1['content']}', later: '{msg2['content']}')"
            response_parts.append(conflict_msg)
            
        return " ".join(response_parts)
