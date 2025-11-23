# tests/conftest.py
import pytest
import os
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_data():
    """Sample tweet data for testing"""
    return pd.DataFrame({
        'content': [
            'I love this beautiful day',
            'I am so happy and grateful',
            'This is terrible and sad',
            'I feel so depressed today',
            'Amazing wonderful fantastic',
            'Horrible awful disappointing'
        ],
        'sentiment': [1, 1, 0, 0, 1, 0]
    })

@pytest.fixture
def sample_raw_tweets():
    """Raw unprocessed tweets"""
    return pd.DataFrame({
        'tweet_id': [1, 2, 3, 4],
        'content': [
            'I LOVE this!!! http://example.com',
            'So sad 😢 123',
            'Amazing day!',
            'Terrible news...'
        ],
        'sentiment': ['happiness', 'sadness', 'happiness', 'sadness']
    })

@pytest.fixture
def preprocessor():
    """Text preprocessor instance"""
    from app import TextPreprocessor
    return TextPreprocessor()

@pytest.fixture
def mock_model():
    """Mock trained model"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    return model

@pytest.fixture
def mock_vectorizer(sample_data):
    """Mock TF-IDF vectorizer"""
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    vectorizer.fit(sample_data['content'])
    return vectorizer

@pytest.fixture
def test_params():
    """Test parameters - Load from actual params.yaml"""
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        # Fallback params
        return {
            'data_ingestion': {
                'test_size': 0.2,
                'random_state': 42
            },
            'data_preprocessing': {
                'lowercase': True,
                'remove_urls': True,
                'remove_punctuations': True,
                'remove_numbers': True,
                'remove_stopwords': True,
                'apply_lemmatization': True,
                'min_words_per_tweet': 2
            },
            'feature_engineering': {
                'max_features': 3000,
                'ngram_range': [1, 2]
            },
            'model_building': {
                'n_estimators': 300,
                'max_depth': 6,
                'random_state': 41,
                'n_jobs': -1
            }
        }

@pytest.fixture
def flask_app():
    """Flask app for testing"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from app import app
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(flask_app):
    """Flask test client"""
    return flask_app.test_client()