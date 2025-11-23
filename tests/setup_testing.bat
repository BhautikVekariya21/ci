@echo off
echo Installing testing dependencies...
pip install pytest pytest-cov pytest-html pytest-flask coverage

echo.
echo Testing dependencies installed!
echo.
echo You can now run:
echo   pytest
echo   pytest --cov=src --cov-report=html
echo   python run_tests.py
echo.
pause