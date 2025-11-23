# tests/test_utils.py
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_params_file_exists():
    """Test that params.yaml exists"""
    assert os.path.exists('params.yaml'), "params.yaml not found"


def test_load_params():
    """Test loading parameters"""
    from utils import load_params
    
    params = load_params()
    assert params is not None, "Params is None"
    assert isinstance(params, dict), "Params should be dictionary"


def test_params_structure():
    """Test params.yaml structure"""
    from utils import load_params
    
    params = load_params()
    
    required_sections = [
        'data_ingestion',
        'data_preprocessing',
        'feature_engineering',
        'model_building',
        'model_evaluation'
    ]
    
    for section in required_sections:
        assert section in params, f"Section {section} missing from params"


def test_logger():
    """Test logger functionality"""
    from utils import logger
    
    assert logger is not None, "Logger is None"
    
    # Test logging
    try:
        logger.info("Test log message")
    except Exception as e:
        pytest.fail(f"Logger failed: {e}")