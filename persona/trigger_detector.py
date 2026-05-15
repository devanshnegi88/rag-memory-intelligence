"""Trigger Detector module for identifying causes of persona drift."""
from typing import List, Dict
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class TriggerDetector:
    """
    Identifies potential triggers (topics, events, persons) in messages.
    """
    def __init__(self):
        # Common stop words to filter out
        self.stop_words = {
            'the', 'and', 'a', 'to', 'of', 'i', 'in', 'is', 'that', 'it', 'on', 'you', 'for', 
            'with', 'was', 'as', 'at', 'be', 'this', 'have', 'from', 'my', 'me', 'hi', 'hello'
        }

    def detect_triggers(self, messages: List[Dict], top_k: int = 3) -> List[str]:
        """
        Detects significant topics/keywords in a set of messages.
        """
        all_text = " ".join([m['content'] for m in messages]).lower()
        
        # 1. Look for explicit "trigger" phrases
        trigger_phrases = [
            r"because of ([^.!?\n]+)",
            r"due to ([^.!?\n]+)",
            r"about ([^.!?\n]+)",
            r"excited for ([^.!?\n]+)",
            r"frustrated with ([^.!?\n]+)",
            r"dealing with ([^.!?\n]+)"
        ]
        
        explicit_triggers = []
        for pattern in trigger_phrases:
            matches = re.finditer(pattern, all_text)
            for match in matches:
                phrase = match.group(1).strip()
                # Take the first few words
                words = phrase.split()[:3]
                if words:
                    explicit_triggers.append(" ".join(words))

        if explicit_triggers:
            return list(set(explicit_triggers))[:top_k]

        # 2. Fallback: Frequent nouns/keywords
        # Very basic approach: extract words, filter stop words, count
        words = re.findall(r'\b[a-z]{4,}\b', all_text) # Only words with length > 3
        filtered_words = [w for w in words if w not in self.stop_words]
        
        counts = Counter(filtered_words)
        return [word for word, count in counts.most_common(top_k)]
