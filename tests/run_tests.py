# run_tests.py
import pytest
import sys

if __name__ == "__main__":
    # Run all tests with verbose output
    exit_code = pytest.main([
        'tests/',
        '-v',
        '--tb=short',
        '--color=yes',
        '--cov=src',
        '--cov-report=html',
        '--cov-report=term'
    ])
    
    sys.exit(exit_code)