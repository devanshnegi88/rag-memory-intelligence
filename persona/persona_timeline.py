"""Persona Timeline module for tracking changes over time."""
import pandas as pd
import json
from typing import List, Dict
from datetime import datetime
import os
import logging
from .extractor import PersonaExtractor
from .drift_detector import DriftDetector
from .trigger_detector import TriggerDetector

logger = logging.getLogger(__name__)

class PersonaTimeline:
    """
    Orchestrates daily persona extraction and tracks changes over time.
    """
    def __init__(self, conversations_path: str, output_dir: str):
        self.conversations_path = conversations_path
        self.output_dir = output_dir
        self.extractor = PersonaExtractor()
        self.drift_detector = DriftDetector()
        self.trigger_detector = TriggerDetector()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_timeline(self, user_id: str = "User 1", max_rows: int = None) -> List[Dict]:
        """
        Reads conversations, groups by date, extracts persona daily, and detects drift.
        """
        df = pd.read_csv(self.conversations_path)
        if max_rows:
            df = df.head(max_rows)
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        daily_personas = []
        
        # Group by date
        grouped = df.groupby('date')
        
        prev_persona = None
        
        for date, group in grouped:
            date_str = date.strftime('%Y-%m-%d')
            
            # Combine all conversations for that day for the specific user
            # In the CSV, conversation is a block of text with 'User 1: ... \n User 2: ...'
            user_messages = []
            msg_idx = 0
            for _, row in group.iterrows():
                conv_text = row['conversation']
                # Basic parsing: split by line and look for the user's name
                lines = conv_text.split('\n')
                for line in lines:
                    if line.startswith(f"{user_id}:"):
                        content = line[len(user_id)+1:].strip()
                        user_messages.append({
                            'content': content,
                            'message_index': msg_idx
                        })
                        msg_idx += 1
            
            if not user_messages:
                continue
                
            # Extract persona for this day
            current_persona = self.extractor.extract(user_messages)
            
            # Detect triggers for this day's content
            triggers = self.trigger_detector.detect_triggers(user_messages)
            
            daily_state = {
                "day": date_str,
                "persona": [trait['trait'] for trait in current_persona['personality_traits']],
                "formality": current_persona['communication_style']['formality'],
                "sentiment": "positive" if any(t['trait'] == 'positive sentiment' for t in current_persona['personality_traits']) else "neutral",
                "trigger": ", ".join(triggers) if triggers else "None"
            }
            
            # Detect drift if we have a previous day
            if prev_persona:
                drift = self.drift_detector.calculate_drift(prev_persona, current_persona)
                daily_state["drift_score"] = drift
            
            daily_personas.append(daily_state)
            prev_persona = current_persona
            
        # Save timeline
        timeline_path = os.path.join(self.output_dir, "persona_timeline.json")
        with open(timeline_path, 'w') as f:
            json.dump(daily_personas, f, indent=2)
            
        return daily_personas
