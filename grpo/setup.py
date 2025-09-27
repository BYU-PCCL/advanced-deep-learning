"""
Run this script ONLY on the login node (not a compute node).
It downloads model weights and datasets needed for training.
"""

import socket

def check_internet(host="huggingface.co", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except Exception:
        return False

if not check_internet():
    print("WARNING: No internet connectivity. Downloads will fail. Run this on the login node with internet access.")
else:
    # Download model weights and tokenizer
    from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B')
    from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')

    # Download GSM8K dataset
    from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='train')