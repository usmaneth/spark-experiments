import ctypes, os, sys
import numpy as np

# Load libllama
lib_path = "/home/usman/Bonsai-demo/bin/cuda/libllama.so"
os.environ["LD_LIBRARY_PATH"] = "/home/usman/Bonsai-demo/bin/cuda"
lib = ctypes.CDLL(lib_path)

print("Loaded libllama successfully!")
