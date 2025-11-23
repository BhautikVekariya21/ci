# tests/test_model.py
import pytest
import pickle
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

class TestModel:
    """Test model functionality"""
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        assert os.path.exists('models/model.pkl'), "Model file not found"
    
    def test_model_loads(self):
        """Test that model can be loaded"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Model is None"
    
    def test_model_type(self):
        """Test model type"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, RandomForestClassifier), "Wrong model type"
    
    def test_model_has_required_methods(self):
        """Test that model has required methods"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        assert hasattr(model, 'predict'), "Model missing predict method"
        assert hasattr(model, 'predict_proba'), "Model missing predict_proba method"
        assert hasattr(model, 'fit'), "Model missing fit method"
    
    def test_model_prediction_shape(self):
        """Test prediction output shape"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test features
        test_feat = pd.read_csv('data/features/test_bow.csv')
        X_test = test_feat.drop('label', axis=1).values[:10]  # Test on 10 samples
        
        predictions = model.predict(X_test)
        
        assert predictions.shape[0] == 10, "Wrong prediction shape"
        assert predictions.ndim == 1, "Predictions should be 1D"
    
    def test_model_prediction_values(self):
        """Test prediction values are binary"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        test_feat = pd.read_csv('data/features/test_bow.csv')
        X_test = test_feat.drop('label', axis=1).values[:10]
        
        predictions = model.predict(X_test)
        
        unique_preds = np.unique(predictions)
        assert set(unique_preds).issubset({0, 1}), "Predictions should be 0 or 1"
    
    def test_model_probabilities(self):
        """Test prediction probabilities"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        test_feat = pd.read_csv('data/features/test_bow.csv')
        X_test = test_feat.drop('label', axis=1).values[:10]
        
        probabilities = model.predict_proba(X_test)
        
        # Should have 2 columns (binary classification)
        assert probabilities.shape[1] == 2, "Should have 2 probability columns"
        
        # Probabilities should sum to 1
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities don't sum to 1"
        
        # All probabilities should be between 0 and 1
        assert (probabilities >= 0).all() and (probabilities <= 1).all(), "Invalid probability values"
    
    def test_model_parameters(self):
        """Test model has correct parameters from params.yaml"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load actual params
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)['model_building']
        
        assert model.n_estimators == params['n_estimators'], f"Expected {params['n_estimators']}, got {model.n_estimators}"
        assert model.random_state == params['random_state'], f"Expected {params['random_state']}, got {model.random_state}"


def test_model_consistency():
    """Test model produces consistent predictions"""
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    test_feat = pd.read_csv('data/features/test_bow.csv')
    X_test = test_feat.drop('label', axis=1).values[:5]
    
    pred1 = model.predict(X_test)
    pred2 = model.predict(X_test)
    
    assert np.array_equal(pred1, pred2), "Model predictions not consistent"


def test_model_feature_importance():
    """Test feature importance"""
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    assert hasattr(model, 'feature_importances_'), "No feature importances"
    
    importances = model.feature_importances_
    
    # Should have importance for each feature
    test_feat = pd.read_csv('data/features/test_bow.csv')
    n_features = test_feat.shape[1] - 1  # Exclude label
    
    assert len(importances) == n_features, "Wrong number of feature importances"
    
    # Importances should sum to 1
    assert np.isclose(importances.sum(), 1.0), "Feature importances don't sum to 1"