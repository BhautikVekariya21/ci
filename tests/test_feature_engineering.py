# tests/test_feature_engineering.py
import pytest
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_feature_files_exist(self):
        """Test that feature files are created"""
        assert os.path.exists('data/features/train_bow.csv'), "Training features not found"
        assert os.path.exists('data/features/test_bow.csv'), "Test features not found"
    
    def test_feature_dimensions(self):
        """Test feature dimensions"""
        train_feat = pd.read_csv('data/features/train_bow.csv')
        test_feat = pd.read_csv('data/features/test_bow.csv')
        
        # Should have same number of features
        assert train_feat.shape[1] == test_feat.shape[1], "Feature dimension mismatch"
    
    def test_label_column_exists(self):
        """Test that label column exists"""
        train_feat = pd.read_csv('data/features/train_bow.csv')
        
        assert 'label' in train_feat.columns, "Label column missing"
    
    def test_feature_values_range(self):
        """Test that TF-IDF values are in valid range"""
        train_feat = pd.read_csv('data/features/train_bow.csv')
        
        # Remove label column
        features = train_feat.drop('label', axis=1)
        
        # TF-IDF values should be between 0 and 1
        assert features.min().min() >= 0, "Negative TF-IDF values found"
        assert features.max().max() <= 1, "TF-IDF values > 1 found"
    
    def test_no_missing_values(self):
        """Test for missing values in features"""
        train_feat = pd.read_csv('data/features/train_bow.csv')
        
        assert train_feat.isna().sum().sum() == 0, "Missing values in features"
    
    def test_vectorizer_consistency(self, sample_data):
        """Test vectorizer produces consistent results"""
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        
        X1 = vectorizer.fit_transform(sample_data['content'])
        X2 = vectorizer.transform(sample_data['content'])
        
        # Should produce same output for same input
        assert np.allclose(X1.toarray(), X2.toarray()), "Vectorizer not consistent"
    
    def test_max_features_respected(self):
        """Test that max_features parameter is respected"""
        train_feat = pd.read_csv('data/features/train_bow.csv')
        
        # Load actual params from params.yaml
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        # Subtract 1 for label column
        n_features = train_feat.shape[1] - 1
        max_features = params['feature_engineering']['max_features']
        
        assert n_features <= max_features, f"Too many features: {n_features} > {max_features}"


def test_sparse_matrix_handling(sample_data):
    """Test handling of sparse matrices"""
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(sample_data['content'])
    
    # Should be sparse
    assert hasattr(X, 'toarray'), "Not a sparse matrix"
    
    # Convert to dense
    X_dense = X.toarray()
    assert isinstance(X_dense, np.ndarray), "Failed to convert to dense"