# tests/test_data_ingestion.py
import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataIngestion:
    """Test data ingestion functionality"""
    
    def test_data_files_exist(self):
        """Test that data files are created"""
        assert os.path.exists('data/raw/train.csv'), "Training data not found"
        assert os.path.exists('data/raw/test.csv'), "Test data not found"
    
    def test_data_loading(self):
        """Test loading data from CSV"""
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        
        assert not train_df.empty, "Training data is empty"
        assert not test_df.empty, "Test data is empty"
    
    def test_data_columns(self):
        """Test that data has required columns"""
        train_df = pd.read_csv('data/raw/train.csv')
        
        required_columns = ['content', 'sentiment']
        for col in required_columns:
            assert col in train_df.columns, f"Column {col} missing"
    
    def test_sentiment_values(self):
        """Test sentiment values are binary"""
        train_df = pd.read_csv('data/raw/train.csv')
        
        unique_sentiments = train_df['sentiment'].unique()
        assert set(unique_sentiments).issubset({0, 1}), "Sentiments should be 0 or 1"
    
    def test_train_test_split_ratio(self, test_params):
        """Test train-test split ratio"""
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        
        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        expected_ratio = test_params['data_ingestion']['test_size']
        
        # Allow 5% tolerance
        assert abs(test_ratio - expected_ratio) < 0.05, f"Test ratio {test_ratio} != {expected_ratio}"
    
    def test_no_missing_values(self):
        """Test for missing values in critical columns"""
        train_df = pd.read_csv('data/raw/train.csv')
        
        assert train_df['content'].isna().sum() == 0, "Missing values in content"
        assert train_df['sentiment'].isna().sum() == 0, "Missing values in sentiment"
    
    def test_data_types(self):
        """Test data types"""
        train_df = pd.read_csv('data/raw/train.csv')
        
        assert train_df['content'].dtype == 'object', "Content should be string"
        assert train_df['sentiment'].dtype in ['int64', 'int32'], "Sentiment should be integer"


def test_stratified_split():
    """Test that train-test split is stratified"""
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    train_ratio = train_df['sentiment'].mean()
    test_ratio = test_df['sentiment'].mean()
    
    # Ratios should be similar (within 5%)
    assert abs(train_ratio - test_ratio) < 0.05, "Split not properly stratified"