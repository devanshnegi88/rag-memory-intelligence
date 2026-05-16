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
        
        # Track attributes of entities found in messages
        # Structure: { (entity_type, entity_name): (value, message_dict) }
        # e.g., { ("sister", "job"): ("doctor", msg_dict) }
        attributes = {}
        
        # Sort results by index to ensure chronological comparison
        # (Assuming message_index or date exists)
        sorted_results = sorted(results, key=lambda x: x.get('message_index', 0))
        
        for res in sorted_results:
            content = res['content'].lower()
            
            # 1. Detect Relationship Attributes (e.g., "My sister is a doctor")
            # Patterns: "my [relation] is/works as/lives in [value]"
            rel_patterns = [
                (r"my (sister|brother|friend|mom|dad|boss) (?:is|works as) a? (\w+)", "job"),
                (r"my (sister|brother|friend|mom|dad|boss) lives in (\w+)", "location"),
                (r"my (sister|brother|friend|mom|dad|boss) is (\d+) years old", "age")
            ]
            
            for pattern, attr_type in rel_patterns:
                match = self._extract_groups(content, pattern)
                if match:
                    relation, value = match
                    key = (relation, attr_type)
                    
                    if key in attributes:
                        old_value, old_msg = attributes[key]
                        if old_value != value:
                            conflicts.append((old_msg, res, f"{relation.capitalize()}'s {attr_type} changed: {old_value} -> {value}"))
                    
                    attributes[key] = (value, res)
            
            # 2. General Self-Attributes (e.g., "I live in Delhi")
            self_patterns = [
                (r"i (?:live in|moved to) (\w+)", "residence"),
                (r"i (?:am|work as) a? (\w+)", "job")
            ]
            
            for pattern, attr_type in self_patterns:
                value = self._extract_pattern(content, pattern)
                if value:
                    key = ("self", attr_type)
                    if key in attributes:
                        old_value, old_msg = attributes[key]
                        if old_value != value:
                            conflicts.append((old_msg, res, f"Personal {attr_type} changed: {old_value} -> {value}"))
                    attributes[key] = (value, res)
                
        return conflicts

    def _extract_groups(self, text: str, pattern: str) -> Tuple[str, ...]:
        import re
        match = re.search(pattern, text)
        if match:
            return match.groups()
        return None

    def _extract_pattern(self, text: str, pattern: str) -> str:
        import re
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None
