"""
setup_confident_ai.py
---------------------
Run this once after cloning the repo to register your Confident AI API key
with DeepEval so evaluation results are pushed to the dashboard automatically.

Usage:
    source venv/bin/activate
    python3 setup_confident_ai.py

Prerequisites:
    1. Add CONFIDENT_AI_API_KEY to your .env file
       Get your key from: https://app.confident-ai.com → Settings → API Keys
"""

import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("CONFIDENT_AI_API_KEY")

if not key or key == "your_confident_ai_api_key_here":
    print()
    print("❌ CONFIDENT_AI_API_KEY not found or not set in .env")
    print()
    print("Steps:")
    print("  1. Go to https://app.confident-ai.com")
    print("  2. Sign up / log in → create a project")
    print("  3. Go to Settings → API Keys → copy your key")
    print("  4. Add to .env:  CONFIDENT_AI_API_KEY=your_key_here")
    print("  5. Re-run: python3 setup_confident_ai.py")
    print()
    exit(1)

try:
    from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues

    KEY_FILE_HANDLER.write_key(KeyValues.CONFIDENT_API_KEY, key)

    # Verify it was written
    stored = KEY_FILE_HANDLER.fetch_data(KeyValues.CONFIDENT_API_KEY)
    if stored == key:
        print()
        print("✅ Confident AI API key registered successfully!")
        print(f"   Key: {key[:8]}...{key[-4:]}")
        print()
        print("When you run evaluations, results will automatically appear at:")
        print("   https://app.confident-ai.com → Testing → Test Runs")
        print()
    else:
        print("⚠️  Key may not have been stored correctly. Check DeepEval config.")

except Exception as e:
    print(f"❌ Error registering key: {e}")
    print()
    print("Try setting it via environment variable instead:")
    print("  Add to .env:  CONFIDENT_API_KEY=your_key_here")
