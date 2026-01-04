#!/bin/bash
# Quick start script for AI Assistant Usage Behavior Analysis

echo "========================================"
echo "AI Assistant Usage Behavior Analysis"
echo "========================================"
echo ""
echo "Choose an option:"
echo "1. Run command-line analysis"
echo "2. Launch GUI application"
echo "3. Install dependencies"
echo "4. Run tests"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running command-line analysis..."
        python main.py
        ;;
    2)
        echo ""
        echo "Launching GUI application..."
        python gui.py
        ;;
    3)
        echo ""
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo "Dependencies installed successfully!"
        ;;
    4)
        echo ""
        echo "Running tests..."
        python -c "from analysis import AIUsageBehaviorAnalysis; print('Analysis module: OK')"
        python -c "import wx; from gui import MainFrame; print('GUI module: OK')"
        python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies: OK')"
        echo "All tests passed!"
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-4."
        exit 1
        ;;
esac
