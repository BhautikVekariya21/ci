# tests/test_preprocessing.py
import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import TextPreprocessor

class TestTextPreprocessor:
    """Test text preprocessing functions"""
    
    def test_clean_text_lowercase(self, preprocessor):
        """Test lowercase conversion in clean_text"""
        text = "HELLO WORLD"
        result = preprocessor.clean_text(text)
        assert result.islower() or result == "", "Text not lowercase"
    
    def test_clean_text_removes_urls(self, preprocessor):
        """Test URL removal in clean_text"""
        text = "Check this http://example.com and www.test.com"
        result = preprocessor.clean_text(text)
        assert "http://" not in result, "HTTP URL not removed"
        assert "www." not in result, "WWW URL not removed"
    
    def test_clean_text_removes_punctuations(self, preprocessor):
        """Test punctuation removal in clean_text"""
        text = "Hello, world! How are you?"
        result = preprocessor.clean_text(text)
        assert "," not in result, "Comma not removed"
        assert "!" not in result, "Exclamation not removed"
        assert "?" not in result, "Question mark not removed"
    
    def test_clean_text_removes_numbers(self, preprocessor):
        """Test number removal in clean_text"""
        text = "I have 123 apples and 456 oranges"
        result = preprocessor.clean_text(text)
        assert "123" not in result, "Numbers not removed"
        assert "456" not in result, "Numbers not removed"
    
    def test_clean_text_removes_stopwords(self, preprocessor):
        """Test stopword removal in clean_text"""
        text = "this is a test of stopwords"
        result = preprocessor.clean_text(text)
        # After stopword removal, common words should be gone
        # But content words should remain
        assert len(result) > 0, "All text removed"
    
    def test_clean_text_empty_input(self, preprocessor):
        """Test empty input handling"""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   ") == ""
        assert preprocessor.clean_text(None) == ""
    
    def test_clean_text_full_pipeline(self, preprocessor):
        """Test full cleaning pipeline"""
        text = "I LOVE this!!! http://example.com 123"
        result = preprocessor.clean_text(text)
        
        assert result.islower() or result == "", "Not lowercase"
        assert "http://" not in result, "URL not removed"
        assert "123" not in result, "Numbers not removed"
        assert "!" not in result, "Punctuation not removed"
    
    def test_clean_text_preserves_content(self, preprocessor):
        """Test that meaningful words are preserved"""
        text = "happy wonderful day"
        result = preprocessor.clean_text(text)
        
        # At least some content should remain
        assert len(result) > 0, "All content removed"
    
    def test_clean_text_consistency(self, preprocessor):
        """Test that clean_text is consistent"""
        text = "This is a test message"
        result1 = preprocessor.clean_text(text)
        result2 = preprocessor.clean_text(text)
        
        assert result1 == result2, "Inconsistent results"


def test_processed_data_exists():
    """Test that processed data files exist"""
    assert os.path.exists('data/processed/train_processed.csv'), "Processed training data not found"
    assert os.path.exists('data/processed/test_processed.csv'), "Processed test data not found"


def test_processed_data_quality():
    """Test quality of processed data"""
    train_df = pd.read_csv('data/processed/train_processed.csv')
    
    # Check no empty strings
    empty_count = (train_df['content'] == '').sum()
    assert empty_count == 0, f"{empty_count} empty strings found"
    
    # Check all lowercase
    uppercase_count = train_df['content'].str.contains('[A-Z]', na=False).sum()
    assert uppercase_count == 0, f"{uppercase_count} uppercase strings found"


def test_min_words_filter():
    """Test minimum words filter"""
    train_df = pd.read_csv('data/processed/train_processed.csv')
    
    word_counts = train_df['content'].str.split().str.len()
    min_words = word_counts.min()
    
    assert min_words >= 2, f"Minimum word count is {min_words}, expected >= 2"