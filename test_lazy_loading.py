#!/usr/bin/env python3
"""
Test script to verify lazy loading is working correctly.
This script should complete quickly without loading heavy ML libraries.
"""
import sys
import time

def test_lazy_loading():
    """Test that importing app modules doesn't load heavy dependencies."""

    print("=" * 70)
    print("LAZY LOADING TEST")
    print("=" * 70)

    # Track heavy modules
    heavy_modules = ['torch', 'transformers', 'faster_whisper', 'ctranslate2', 'pyannote']

    # Initial state
    print("\n1. Initial state:")
    print(f"   Total modules loaded: {len(sys.modules)}")
    for mod in heavy_modules:
        present = mod in sys.modules
        print(f"   {mod:20} loaded: {present}")

    # Import app config (should be fast and lightweight)
    print("\n2. After importing app_config:")
    start = time.time()
    from whisperx.app.app_config import AppConfig, TranscriptionConfig
    elapsed = time.time() - start
    print(f"   Import time: {elapsed:.3f} seconds")
    print(f"   Total modules loaded: {len(sys.modules)}")
    for mod in heavy_modules:
        present = mod in sys.modules
        status = "❌ FAILED" if present else "✓ OK"
        print(f"   {mod:20} loaded: {present:5} {status}")

    # Verify no heavy modules loaded
    heavy_loaded = [mod for mod in heavy_modules if mod in sys.modules]

    print("\n" + "=" * 70)
    if heavy_loaded:
        print("❌ LAZY LOADING FAILED!")
        print(f"   Heavy modules loaded at import time: {', '.join(heavy_loaded)}")
        print("   App will have slow startup.")
        return False
    else:
        print("✓ LAZY LOADING WORKING!")
        print("   No heavy ML libraries loaded at import time.")
        print("   App will have fast startup.")
        print(f"   Import completed in {elapsed:.3f} seconds")
        return True

if __name__ == "__main__":
    success = test_lazy_loading()
    sys.exit(0 if success else 1)
