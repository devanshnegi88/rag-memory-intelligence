"""Drift Detector module for comparing persona states."""
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects changes (drift) between two persona states.
    """
    def calculate_drift(self, persona_a: Dict, persona_b: Dict) -> float:
        """
        Calculates a drift score (0 to 1) between two persona states.
        Focuses on changes in communication style and personality traits.
        """
        score = 0.0
        
        # 1. Compare Formality
        if persona_a['communication_style']['formality'] != persona_b['communication_style']['formality']:
            score += 0.3
            
        # 2. Compare Verbosity
        if persona_a['communication_style']['verbosity'] != persona_b['communication_style']['verbosity']:
            score += 0.2
            
        # 3. Compare Traits (Set difference)
        traits_a = set([t['trait'] for t in persona_a['personality_traits']])
        traits_b = set([t['trait'] for t in persona_b['personality_traits']])
        
        if traits_a or traits_b:
            intersection = traits_a.intersection(traits_b)
            union = traits_a.union(traits_b)
            # Jaccard distance for traits
            jaccard_dist = 1.0 - (len(intersection) / len(union)) if union else 0.0
            score += jaccard_dist * 0.5
            
        return min(score, 1.0)

    def identify_drift_reasons(self, persona_a: Dict, persona_b: Dict) -> List[str]:
        """
        Identifies specific reasons for detected drift.
        """
        reasons = []
        if persona_a['communication_style']['formality'] != persona_b['communication_style']['formality']:
            reasons.append(f"Formality changed from {persona_a['communication_style']['formality']} to {persona_b['communication_style']['formality']}")
            
        traits_a = set([t['trait'] for t in persona_a['personality_traits']])
        traits_b = set([t['trait'] for t in persona_b['personality_traits']])
        
        added = traits_b - traits_a
        removed = traits_a - traits_b
        
        for t in added:
            reasons.append(f"Gained trait: {t}")
        for t in removed:
            reasons.append(f"Lost trait: {t}")
            
        return reasons
