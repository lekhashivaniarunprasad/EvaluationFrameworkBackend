#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — First-time setup for LLM Evaluation Platform
# Run this once after cloning the repo:
#   chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "🚀 LLM Evaluation Platform — First Time Setup"
echo "================================================"

# ── Step 1: Create .env from example ─────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
    echo "⚠️  Please fill in your values in .env before continuing"
else
    echo "✅ .env already exists — skipping"
fi

# ── Step 2: Create virtual environment ───────────────────────────────────────
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists — skipping"
fi

# ── Step 3: Install dependencies ─────────────────────────────────────────────
echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"

# ── Step 4: Set up Confident AI ──────────────────────────────────────────────
echo ""
echo "🔑 Setting up Confident AI (DeepEval Dashboard)..."
echo ""
echo "   To use the Confident AI dashboard:"
echo "   1. Go to https://app.confident-ai.com and create a free account"
echo "   2. Create a project and copy your API key from Settings → API Keys"
echo "   3. Add it to your .env file: CONFIDENT_AI_API_KEY=your_key_here"
echo "   4. Run the command below to register it with DeepEval:"
echo ""
echo "      source venv/bin/activate"
echo "      python3 -c \""
echo "      from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues"
echo "      import os"
echo "      from dotenv import load_dotenv"
echo "      load_dotenv()"
echo "      key = os.getenv('CONFIDENT_AI_API_KEY')"
echo "      if key:"
echo "          KEY_FILE_HANDLER.write_key(KeyValues.CONFIDENT_API_KEY, key)"
echo "          print('✅ Confident AI key registered successfully')"
echo "      else:"
echo "          print('❌ CONFIDENT_AI_API_KEY not found in .env')"
echo "      \""
echo ""

# ── Step 5: Auto-register Confident AI key if already in .env ────────────────
if [ -f ".env" ]; then
    source venv/bin/activate
    python3 - << 'PYEOF'
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("CONFIDENT_AI_API_KEY")
if key and key != "your_confident_ai_api_key_here":
    try:
        from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
        KEY_FILE_HANDLER.write_key(KeyValues.CONFIDENT_API_KEY, key)
        print("✅ Confident AI API key registered automatically from .env")
    except Exception as e:
        print(f"⚠️  Could not register Confident AI key: {e}")
        print("   Run the manual command above after filling in your .env")
else:
    print("⚠️  CONFIDENT_AI_API_KEY not set in .env yet")
    print("   Add your key to .env and re-run: python3 setup_confident_ai.py")
PYEOF
fi

echo ""
echo "================================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Fill in all values in .env"
echo "  2. Start the server:"
echo "     source venv/bin/activate"
echo "     uvicorn main:app --reload --port 8000"
echo ""
