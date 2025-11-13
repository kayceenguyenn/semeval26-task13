#!/bin/bash
# Environment setup script for SemEval 2026 Task 13
# Ensures reproducible environment across all machines

set -e  # Exit on error

echo "🔧 SemEval 2026 Task 13 - Environment Setup"
echo "==========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✅ Python $PYTHON_VERSION detected (>= 3.9 required)"
else
    echo "❌ ERROR: Python 3.9+ required, found $PYTHON_VERSION"
    echo ""
    echo "Please install Python 3.9 or higher:"
    echo "  - macOS: brew install python@3.9"
    echo "  - Ubuntu: sudo apt install python3.9"
    echo "  - Windows: Download from python.org"
    exit 1
fi

echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf venv
    else
        echo "✅ Using existing virtual environment"
        source venv/bin/activate
        echo "✅ Environment activated!"
        echo ""
        echo "To activate in the future, run:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing dependencies (this may take 2-3 minutes)..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Generate data: python3 src/generate_data.py --task A"
echo "  3. Train model: python3 src/pipeline.py train --task A"
echo "  4. Run tests: ./run_all_tests.sh"
echo ""
echo "📝 To deactivate: deactivate"
echo ""
