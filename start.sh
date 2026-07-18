#!/bin/bash
set -e

echo ""
echo " ╔══════════════════════════════════════════╗"
echo " ║         GAN Studio — Auto Setup          ║"
echo " ║   Image Generator + Text Generator       ║"
echo " ╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " [ERROR] python3 not found. Install from https://python.org"
    exit 1
fi
echo " [1/4] Python found ✓"

# Virtual env
if [ ! -d "venv" ]; then
    echo " [2/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo " [2/4] Virtual environment exists ✓"
fi

# Activate + install
source venv/bin/activate
echo ""
echo " [3/4] Installing dependencies..."
pip install -r backend/requirements.txt -q

echo ""
echo " [4/4] Launching GAN Studio (backend + frontend)..."
echo ""
echo " ════════════════════════════════════════════"
echo "  Open  →  http://localhost:5000"
echo " ════════════════════════════════════════════"
echo ""

cd backend
python app.py