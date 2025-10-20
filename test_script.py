try:
    import torch

    print(f"✅ PyTorch version: {torch.__version__}")

    import whisperx

    print("✅ WhisperX imported successfully")

    # Test CUDA availability (optional)
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  CUDA not available, will use CPU")

except ImportError as e:
    print(f"❌ Import error: {e}")