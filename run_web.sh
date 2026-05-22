#!/bin/bash

# Exit on error
set -e

echo "=== ESCNet Web Server Setup & Run Script ==="

# 1. Check if we need to recreate the virtual environment (since the current one is from Windows)
if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo "⚠️ Detected a Windows-style virtual environment (.venv) on macOS."
    echo "Recreating virtual environment for macOS..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install/upgrade dependencies
echo "Installing and upgrading dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# 4. Start the web application
echo ""
echo "🚀 Starting ESCNet Web Server..."
echo "🔗 Open your web browser and go to: http://127.0.0.1:8000"
echo ""

python app.py
