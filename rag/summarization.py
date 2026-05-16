"""Summarization engine for topics and message checkpoints."""
import json
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SummarizationEngine:
    """
    Generate summaries for:
    1. Topic segments
    2. 100-message checkpoints
    
    Also generates short labels for topics using keyword extraction.
    """
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', 
                 max_length: int = 150, min_length: int = 50,
                 gemini_api_key: str = None):
        """
        Initialize summarization engine.
        
        Args:
            model_name: Summarization model to use
            max_length: Maximum summary length
            min_length: Minimum summary length
            gemini_api_key: Optional Google Gemini API key
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.summarizer = None
        self.gemini_model = None
        
        # Initialize Gemini if key provided
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
                logger.info("Gemini summarization initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        
        # Try to load transformer pipeline, fallback to extractive summarization
        try:
            from transformers import pipeline
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=-1  # Use CPU
            )
            logger.info(f"Loaded transformer summarizer: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer summarizer: {e}. Using extractive summarization.")
            self.summarizer = None
        
    def summarize_topics(self, topics: List[Dict]) -> List[Dict]:
        """
        Generate summaries for each topic.
        
        Adds 'summary' and 'label' keys to each topic dict.
        
        Args:
            topics: List of topic dicts from TopicDetector
            
        Returns:
            Topics with summaries added
        """
        for topic in topics:
            messages = topic['messages']
            
            # Combine message contents
            combined_text = ' '.join([m['content'] for m in messages])
            
            # Generate summary
            if len(combined_text.split()) > 20:  # Only summarize if enough text
                try:
                    summary = self.summarize_text(combined_text)
                except Exception as e:
                    logger.error(f"Error summarizing topic {topic['topic_id']}: {e}")
                    summary = self._fallback_summary(messages)
            else:
                summary = combined_text
            
            topic['summary'] = summary
            
            # Generate label from keywords
            topic['label'] = self._generate_label(topic['keywords'])
            
        return topics
    
    def create_checkpoint_summaries(self, messages: List[Dict], 
                                    checkpoint_size: int = 100) -> List[Dict]:
        """
        Create summaries at regular intervals (every checkpoint_size messages).
        
        Returns list of checkpoint dicts:
        {
            'checkpoint_id': int,
            'start_idx': int,
            'end_idx': int,
            'message_count': int,
            'summary': str
        }
        
        Args:
            messages: All messages in conversation
            checkpoint_size: Messages per checkpoint (default 100)
            
        Returns:
            List of checkpoint summaries
        """
        checkpoints = []
        checkpoint_id = 0
        
        for start_idx in range(0, len(messages), checkpoint_size):
            end_idx = min(start_idx + checkpoint_size, len(messages))
            checkpoint_messages = messages[start_idx:end_idx]
            
            # Combine message contents
            combined_text = ' '.join([m['content'] for m in checkpoint_messages])
            
            # Generate summary
            if len(combined_text.split()) > 20:
                try:
                    summary = self.summarize_text(combined_text)
                except Exception as e:
                    logger.error(f"Error summarizing checkpoint {checkpoint_id}: {e}")
                    summary = self._fallback_summary(checkpoint_messages)
            else:
                summary = combined_text
            
            checkpoints.append({
                'checkpoint_id': checkpoint_id,
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'message_count': len(checkpoint_messages),
                'summary': summary
            })
            
            checkpoint_id += 1
        
        return checkpoints
    
    def summarize_text(self, text: str, max_length: int = None, min_length: int = None) -> str:
        """
        Summarize text using transformer model, Gemini, or fallback extractive method.
        """
        # Use defaults if not provided
        max_len = max_length if max_length is not None else self.max_length
        min_len = min_length if min_length is not None else self.min_length
        
        # Truncate to avoid token limits for local models
        words = text.split()
        if len(words) > 1024:
            text_truncated = ' '.join(words[:1024])
        else:
            text_truncated = text
        
        # 1. Try Transformer model
        if self.summarizer is not None:
            try:
                result = self.summarizer(text_truncated, max_length=max_len, 
                                        min_length=min_len, do_sample=False)
                return result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}")

        # 2. Try Gemini fallback
        if self.gemini_model is not None:
            # Extract question and messages safely without backslashes in f-string
            lines = text.split('\n')
            question = 'Summarize the text.'
            if 'Question: ' in text:
                for line in lines:
                    if 'Question: ' in line:
                        question = line.split('Question: ')[-1]
                        break
            
            messages_part = text
            if 'Messages:\n' in text:
                messages_part = text.split('Messages:\n')[-1]

            try:
                prompt = (
                    f"You are an assistant describing a user's behavior and statements based on chat logs.\n"
                    f"Answer the following question based ONLY on the provided messages.\n"
                    f"Always use the third person (e.g., 'The user is...', 'They mentioned...').\n"
                    f"IMPORTANT: Do not include speaker labels like 'User 1:' or 'Assistant:' in your response.\n"
                    f"Question: {question}\n\n"
                    f"Messages:\n{messages_part}\n\n"
                    f"Concise Answer:"
                )
                response = self.gemini_model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini summarization failed: {e}")
        
        # 3. Fallback: extractive summarization
        return self._extractive_summary(text)
    
    def _extractive_summary(self, text: str) -> str:
        """
        Extract key sentences for summary (no model required).
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary
        """
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if len(sentences) <= 2:
            return sentences[0] if sentences else text
        
        # Take first and last sentences, plus middle ones
        summary_sentences = []
        summary_sentences.append(sentences[0])
        
        # Add middle sentences (every nth sentence)
        step = max(1, len(sentences) // 3)
        for i in range(step, len(sentences) - 1, step):
            summary_sentences.append(sentences[i])
        
        summary_sentences.append(sentences[-1])
        
        summary = '. '.join(summary_sentences[:4])  # Max 4 sentences
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def _fallback_summary(self, messages: List[Dict]) -> str:
        """
        Create simple summary when model fails.
        
        Uses first few and last few messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Simple summary
        """
        if not messages:
            return "Empty topic"
        
        # Take first 2 and last 2 messages
        sample_messages = messages[:2] + messages[-2:]
        sample_messages = list(set(sample_messages))  # Remove duplicates
        
        summary_parts = []
        for msg in sample_messages:
            content = msg.get('content', '').strip()
            if content:
                summary_parts.append(content[:100])  # First 100 chars
        
        return ' ... '.join(summary_parts)
    
    def _generate_label(self, keywords: List[str]) -> str:
        """
        Generate short label from keywords.
        
        Args:
            keywords: List of keywords
            
        Returns:
            Short label (max 5 words)
        """
        if not keywords:
            return "General Topic"
        
        # Use top 3 keywords, capitalize
        label_words = [kw.capitalize() for kw in keywords[:3]]
        label = ' & '.join(label_words)
        
        return label if label else "General Topic"
    
    def save_summaries(self, topics: List[Dict], output_path: str) -> None:
        """
        Save topic summaries and checkpoints to JSON.
        
        Args:
            topics: List of topics with summaries
            output_path: Path to save JSON
        """
        # Create simplified structure for output
        output_data = {
            'total_topics': len(topics),
            'topics': []
        }
        
        for topic in topics:
            output_data['topics'].append({
                'topic_id': topic['topic_id'],
                'label': topic.get('label', ''),
                'summary': topic.get('summary', ''),
                'start_idx': topic['start_idx'],
                'end_idx': topic['end_idx'],
                'message_count': len(topic.get('messages', [])),
                'keywords': topic.get('keywords', [])
            })
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
