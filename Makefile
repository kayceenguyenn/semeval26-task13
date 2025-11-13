.PHONY: help setup data train test clean validate format info

help:
	@echo "SemEval 2026 Task 13 - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install dependencies with pip"
	@echo "  make data           - Generate sample data"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Train baseline model (Task A)"
	@echo "  make train-all      - Train all models"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-quick     - Run quick smoke tests"
	@echo ""
	@echo "Other:"
	@echo "  make validate       - Validate submission format"
	@echo "  make clean          - Clean generated files"
	@echo "  make format         - Format code with black"
	@echo "  make info           - Show project info"

# Setup with pip
setup:
	@echo "Installing dependencies..."
	pip3 install -r requirements.txt
	@echo "✅ Done! Run 'make data' to generate sample data"

# Generate sample data
data:
	@echo "Generating sample data..."
	python3 src/generate_data.py --task A
	python3 src/generate_data.py --task B
	@echo "✅ Data generated in data/ directory"

# Train baseline model
train:
	@echo "Training baseline model (Task A)..."
	python3 src/pipeline.py train --task A

# Train all models
train-all:
	@echo "Training all models..."
	python3 src/pipeline.py train --task A --model-type logistic_regression
	python3 src/pipeline.py train --task A --model-type random_forest
	python3 src/pipeline.py train --task A --model-type gradient_boosting

# Run all tests
test:
	@echo "Running all tests..."
	./run_all_tests.sh

# Quick smoke test
test-quick:
	@echo "Running quick tests..."
	python3 tests/test_data_loader.py
	python3 tests/test_features.py

# Validate submission format
validate:
	@echo "Validating submission..."
	python3 src/pipeline.py validate results/predictions/task_A_submission.csv

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf logs/*.log
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

# Format code (requires black)
format:
	@echo "Formatting code..."
	black src/ tests/
	@echo "✅ Code formatted!"

# Show project info
info:
	@echo "Project: SemEval 2026 Task 13"
	@echo "Training data: $(shell ls -1 data/train_*.parquet 2>/dev/null | wc -l) files"
	@echo "Models saved: $(shell ls -1 models/*.pkl 2>/dev/null | wc -l) models"
	@echo "Tests: $(shell ls -1 tests/test_*.py 2>/dev/null | wc -l) test files"
