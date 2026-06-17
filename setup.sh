#!/bin/bash

# Setup script for Unix/Linux/Mac users

echo ""
echo "========================================"
echo "Sentiment Analysis Project - Setup"
echo "========================================"
echo ""

echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    echo "Please install Python 3.10+ from python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python: $PYTHON_VERSION"

echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "To launch the app, run:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "The app will open at http://localhost:8501/"
echo ""
