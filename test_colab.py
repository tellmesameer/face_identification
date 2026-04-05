import sys
import platform

print("=== Colab Connection Test ===")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import google.colab
    print("✅ google.colab module found! You are running in a Google Colab environment.")
except ImportError:
    print("❌ google.colab module NOT found. You might not be connected to the Colab backend.")

# Check for GPU (often why people use Colab)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ CUDA (GPU) is not available. You might be on a CPU-only runtime.")
except ImportError:
    print("ℹ️ PyTorch not installed to verify GPU, skipping GPU check.")

try:
    import subprocess
    print("\nSystem info:")
    subprocess.run(["uname", "-a"])
    
    print("\nGPU info (nvidia-smi):")
    subprocess.run(["nvidia-smi"])
except Exception as e:
    pass
