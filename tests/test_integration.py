# tests/test_integration.py
import pytest
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import TextPreprocessor

class TestIntegration:
    """Integration tests for end-to-end pipeline"""
    
    def test_full_pipeline_single_prediction(self):
        """Test full pipeline from raw text to prediction"""
        # Load components
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        train_df = pd.read_csv('data/processed/train_processed.csv')
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        vectorizer.fit(train_df['content'].fillna(''))
        
        preprocessor = TextPreprocessor()
        
        # Test text
        text = "I absolutely love this amazing product!"
        
        # Preprocess - clean_text takes only one argument
        cleaned = preprocessor.clean_text(text)
        
        # Vectorize
        vec = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        # Assertions
        assert prediction in [0, 1], "Invalid prediction"
        assert len(probability) == 2, "Invalid probability shape"
        assert abs(sum(probability) - 1.0) < 0.01, "Probabilities don't sum to 1"
    
    def test_data_flow_consistency(self):
        """Test data consistency through pipeline"""
        # Load data at different stages
        raw_train = pd.read_csv('data/raw/train.csv')
        processed_train = pd.read_csv('data/processed/train_processed.csv')
        features_train = pd.read_csv('data/features/train_bow.csv')
        
        # Number of samples should decrease or stay same
        assert len(processed_train) <= len(raw_train), "Processed has more samples than raw"
        assert len(features_train) <= len(processed_train), "Features has more samples than processed"
    
    def test_sentiment_distribution_consistency(self):
        """Test sentiment distribution is maintained"""
        raw_train = pd.read_csv('data/raw/train.csv')
        features_train = pd.read_csv('data/features/train_bow.csv')
        
        raw_ratio = raw_train['sentiment'].mean()
        feature_ratio = features_train['label'].mean()
        
        # Ratios should be similar (within 10%)
        assert abs(raw_ratio - feature_ratio) < 0.1, "Sentiment distribution changed significantly"


def test_reproducibility():
    """Test that pipeline is reproducible"""
    import yaml
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    assert 'random_state' in params['data_ingestion'], "No random state for data ingestion"
    assert 'random_state' in params['model_building'], "No random state for model building"