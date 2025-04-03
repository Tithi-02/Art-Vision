import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from transformers import pipeline

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

class EmotionAnalyzer:
    def __init__(self):
        # Initialize emotion classification pipeline
        self.emotion_classifier = pipeline("text-classification", 
                                          model="j-hartmann/emotion-english-distilroberta-base", 
                                          return_all_scores=True)
        
        # Emotion to color/mood mapping
        self.emotion_mappings = {
            'joy': {'color_boost': [1.2, 1.1, 0.9], 'prompt_prefix': 'bright, cheerful'},
            'sadness': {'color_boost': [0.8, 0.8, 1.1], 'prompt_prefix': 'melancholic, somber'},
            'anger': {'color_boost': [1.3, 0.8, 0.8], 'prompt_prefix': 'intense, fiery'},
            'fear': {'color_boost': [0.7, 0.7, 0.8], 'prompt_prefix': 'dark, ominous'},
            'love': {'color_boost': [1.1, 0.8, 1.0], 'prompt_prefix': 'warm, tender'},
            'surprise': {'color_boost': [1.1, 1.1, 0.8], 'prompt_prefix': 'vibrant, dynamic'},
            'neutral': {'color_boost': [1.0, 1.0, 1.0], 'prompt_prefix': 'balanced, neutral'}
        }
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing and removing stopwords"""
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return ' '.join(filtered_tokens)
    
    def analyze_emotion(self, text):
        """Analyze the emotional content of text"""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Get emotion scores
        emotion_scores = self.emotion_classifier(text)[0]
        
        # Find the dominant emotion
        dominant_emotion = max(emotion_scores, key=lambda x: x['score'])
        
        # Get the mapping for the dominant emotion (default to neutral if not found)
        emotion_name = dominant_emotion['label']
        mapping = self.emotion_mappings.get(emotion_name, self.emotion_mappings['neutral'])
        
        result = {
            'dominant_emotion': emotion_name,
            'emotion_score': dominant_emotion['score'],
            'color_boost': mapping['color_boost'],
            'prompt_prefix': mapping['prompt_prefix'],
            'all_emotions': {item['label']: item['score'] for item in emotion_scores}
        }
        
        return result