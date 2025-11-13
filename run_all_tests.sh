#!/bin/bash
# Run all tests

echo "🧪 Running All Tests..."
echo "=" | head -c 70 | tr '\n' '='
echo ""

echo ""
echo "📦 Testing data_loader..."
python3 tests/test_data_loader.py
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "🔧 Testing features..."
python3 tests/test_features.py
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "📊 Testing evaluate..."
python3 tests/test_evaluate.py
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "🚀 Testing pipeline integration..."
python3 tests/test_pipeline.py
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "=" | head -c 70 | tr '\n' '='
echo ""
echo "✅ ALL TESTS PASSED!"
echo "=" | head -c 70 | tr '\n' '='
echo ""
