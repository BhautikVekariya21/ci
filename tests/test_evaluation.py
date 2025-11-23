# tests/test_evaluation.py
import pytest
import json
import os
import pandas as pd

class TestEvaluation:
    """Test model evaluation"""
    
    def test_metrics_file_exists(self):
        """Test that metrics file exists"""
        assert os.path.exists('evaluation/metrics.json'), "Metrics file not found"
    
    def test_metrics_loadable(self):
        """Test that metrics can be loaded"""
        with open('evaluation/metrics.json', 'r') as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    def test_required_metrics_present(self):
        """Test that required metrics are present"""
        with open('evaluation/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            assert metric in metrics, f"Metric {metric} missing"
    
    def test_metrics_in_valid_range(self):
        """Test that metrics are in valid range [0, 1]"""
        with open('evaluation/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} = {value} not in [0, 1]"
    
    def test_metrics_reasonable_values(self):
        """Test that metrics have reasonable values"""
        with open('evaluation/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # For a decent model, accuracy should be > 0.5 (better than random)
        assert metrics['accuracy'] > 0.5, f"Accuracy too low: {metrics['accuracy']}"
    
    def test_confusion_matrix_exists(self):
        """Test that confusion matrix image exists"""
        assert os.path.exists('evaluation/confusion_matrix.png'), "Confusion matrix not found"
    
    def test_classification_report_exists(self):
        """Test that classification report exists"""
        assert os.path.exists('evaluation/classification_report.csv'), "Classification report not found"
    
    def test_classification_report_format(self):
        """Test classification report format"""
        report = pd.read_csv('evaluation/classification_report.csv')
        
        assert not report.empty, "Classification report is empty"
        assert 'precision' in report.columns or 'precision' in report.index, "Precision missing"


def test_precision_recall_relationship():
    """Test relationship between precision and recall"""
    with open('evaluation/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1_score']
    
    # F1 should be harmonic mean of precision and recall
    expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    assert abs(f1 - expected_f1) < 0.01, f"F1 score calculation incorrect"