# src/quantization/test_reconstructed_1bit.py
"""
Test loader & generator for the reconstructed 1-bit state_dict.

Usage:
  python src/quantization/test_reconstructed_1bit.py

Assumes:
  - Base model config/tokenizer dir:
      D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-base
  - Reconstructed state_dict:
      D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-1bit/reconstructed_state_dict.pt
"""
import time
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(r"D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-base")
RECON_STATE = Path(r"D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-1bit/reconstructed_state_dict.pt")

PROMPT = "Explain the environmental impact of plastic waste in one short paragraph."
MAX_NEW_TOKENS = 20
NUM_WARMUP = 1
NUM_RUNS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Device: {device}")

def report_peak_gb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024**3)
    return 0.0

def main():
    assert BASE_DIR.exists(), f"Base dir not found: {BASE_DIR}"
    assert RECON_STATE.exists(), f"Reconstructed state not found: {RECON_STATE}"

    # load tokenizer
    print("[*] Loading tokenizer from base dir...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), use_fast=True)
    print(f"    Tokenizer loaded ({time.time() - t0:.3f}s)")

    # load config and create model object (structure only)
    print("[*] Loading model config and creating model instance (no weights yet)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(str(BASE_DIR))
    model = AutoModelForCausalLM.from_config(config)
    print(f"    Model instance created ({time.time() - t0:.3f}s)")

    # load reconstructed state_dict (might be large)
    print("[*] Loading reconstructed state_dict (this may take a moment)...")
    t0 = time.time()
    state = torch.load(str(RECON_STATE), map_location="cpu")
    print(f"    state_dict loaded ({time.time() - t0:.3f}s), params: {len(state)}")

    # strict=False to allow minor mismatches
    print("[*] Loading state into model (map to device after)...")
    load_start = time.time()
    missing, unexpected = model.load_state_dict(state, strict=False)
    load_time = time.time() - load_start
    print(f"    state_dict -> model done ({load_time:.3f}s)")
    if missing:
        print(f"    WARNING - missing keys: {len(missing)}")
    if unexpected:
        print(f"    WARNING - unexpected keys: {len(unexpected)}")

    # move to device and prepare GPU stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.to(device)
    print(f"[*] Model moved to {device}")

    # warmup
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    print("[*] Warmup generation...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    print("    Warmup done")

    # generation runs - measure times
    gen_times = []
    outputs = []
    with torch.no_grad():
        for i in range(NUM_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            gen_times.append(dt)
            decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            outputs.append(decoded)
            peak = report_peak_gb()
            print(f"Run {i+1}/{NUM_RUNS}: gen_time={dt:.3f}s, gpu_peak_GB={peak:.3f}")

    avg_gen = sum(gen_times) / len(gen_times)
    print("\n=== Summary ===")
    print(f"Load time (state->model): {load_time:.3f}s")
    print(f"Avg generation time ({NUM_RUNS} runs): {avg_gen:.3f}s")
    if device.type == "cuda":
        print(f"Last run GPU peak (GB): {report_peak_gb():.3f}")
    print("\nSample output (full):\n")
    print(outputs[0])

if __name__ == "__main__":
    main()
