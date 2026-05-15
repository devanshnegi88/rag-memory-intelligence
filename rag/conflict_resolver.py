"""Conflict resolver for detecting contradictions in retrieved chunks."""
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ConflictResolver:
    """
    Detects contradictions in retrieved chunks.
    Example: "I live in Delhi" vs "I live in Mumbai".
    """
    def detect_conflicts(self, results: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
        """
        Identifies pairs of results that likely contradict each other.
        Returns a list of (msg1, msg2, reason).
        """
        conflicts = []
        
        # Simple entity-based conflict detection
        # Look for patterns like "My X is Y" vs "My X is Z"
        entities = {} # entity_name -> (value, message_dict)
        
        for res in results:
            content = res['content'].lower()
            
            # Simple patterns
            # 1. "lives in [location]" or "moved to [location]"
            location_match = self._extract_pattern(content, r"(?:lives in|moved to) (\w+)")
            if location_match:
                if "residence" in entities and entities["residence"][0] != location_match:
                    conflicts.append((entities["residence"][1], res, f"Residence conflict: {entities['residence'][0]} vs {location_match}"))
                entities["residence"] = (location_match, res)
                
            # 2. "is a [occupation]"
            job_match = self._extract_pattern(content, r"is a (\w+)")
            if job_match:
                if "job" in entities and entities["job"][0] != job_match:
                    conflicts.append((entities["job"][1], res, f"Job conflict: {entities['job'][0]} vs {job_match}"))
                entities["job"] = (job_match, res)
                
        return conflicts

    def _extract_pattern(self, text: str, pattern: str) -> str:
        import re
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None
