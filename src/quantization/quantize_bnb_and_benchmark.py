# src/quantization/quantize_bnb_and_benchmark.py
"""
Quantize base HF model into 4-bit (NF4) and 8-bit (bnb int8) variants using BitsAndBytesConfig,
save them as local HF-style folders, and measure load/gen/peak metrics.

Usage (from repo root):
  python src/quantization/quantize_bnb_and_benchmark.py

Adjust BASE_DIR and OUT_BASE constants below to your paths.
"""
import time, os, json, gc
from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ---------- USER PATHS (edit if needed) ----------
BASE_DIR = Path(r"D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-base")
OUT_BASE = Path(r"D:/Phi3-EdgeQuant-Agent/models")  # will create subfolders inside
PROMPT = "Explain the environmental impact of plastic waste in one short paragraph."
MAX_NEW_TOKENS = 20
WARMUPS = 2
MEASURE_RUNS = 3

# ---------- helpers ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Running on device: {device}")

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def report_peak_gb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024**3)
    return 0.0

def time_load_and_save(bnb_config: BitsAndBytesConfig, out_folder: Path, label: str):
    """
    Load model with the provided BitsAndBytesConfig, save it to out_folder,
    run a few warmups and measured generations, and return metrics + a sample output.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # load tokenizer first (fast)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR), use_fast=True)
    t_tokenizer = time.time() - t0

    # load model (timed)
    print(f"[{label}] Loading model with quantization config: {bnb_config}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR),
        quantization_config=bnb_config,
        device_map="auto",         # let HF/accelerate place layers
        trust_remote_code=True     # Qwen may require remote code - keep if you used base from Qwen
    )
    load_time = time.time() - t0
    print(f"[{label}] model load_time: {load_time:.3f}s")

    # ensure eval & to half if on cuda (bnb layers may already be optimized)
    model.eval()
    if device.type == "cuda":
        try:
            model.half()
        except Exception:
            pass

    # warmups + measured runs
    inputs = tokenizer(PROMPT, return_tensors="pt").to(next(model.parameters()).device)
    outputs = []
    run_times = []
    peaks = []

    # full sequence of warmups + measure runs
    total_runs = WARMUPS + MEASURE_RUNS
    for i in range(total_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                num_beams=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        run_times.append(dt)
        peak = report_peak_gb()
        peaks.append(peak)
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        outputs.append(decoded)
        print(f"[{label}] run {i+1}/{total_runs} time={dt:.3f}s peak_GB={peak:.3f}")

    # steady index: last measured (MEASURE_RUNS last runs -> choose last)
    steady_idx = total_runs - 1
    steady_time = run_times[steady_idx]
    steady_peak = peaks[steady_idx]
    steady_out = outputs[steady_idx]

    # attempt to save the quantized model folder (transformers will save metadata; bitsandbytes state may be saved differently)
    print(f"[{label}] saving model to {out_folder} ...")
    try:
        # Many bitsandbytes loads keep quant params that save_pretrained can persist.
        model.save_pretrained(str(out_folder))
        tokenizer.save_pretrained(str(out_folder))
        saved = True
    except Exception as e:
        print(f"[{label}] Warning: save_pretrained failed: {e}")
        saved = False

    # cleanup
    try:
        del model
        del tokenizer
    except Exception:
        pass
    clear_cuda()

    return {
        "label": label,
        "load_time_s": float(load_time),
        "tokenizer_load_s": float(t_tokenizer),
        "steady_gen_time_s": float(steady_time),
        "steady_gpu_peak_gb": float(steady_peak),
        "steady_output": steady_out,
        "save_ok": saved
    }

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base dir not found: {BASE_DIR}")

    results = []

    # 4-bit config (NF4 recommended for 4-bit; double quant + float16 compute)
    bnb4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    out4 = OUT_BASE / "Qwen2.5-3B-Instruct-4bit-from-base"
    res4 = time_load_and_save(bnb4, out4, label="4bit")
    results.append(res4)

    # 8-bit config (int8 path)
    bnb8 = BitsAndBytesConfig(load_in_8bit=True)
    out8 = OUT_BASE / "Qwen2.5-3B-Instruct-8bit-from-base"
    res8 = time_load_and_save(bnb8, out8, label="8bit")
    results.append(res8)

    # write simple JSON summary
    summary_path = OUT_BASE / "quantization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done. Summary written to", summary_path)
    for r in results:
        print("\n", r)

if __name__ == "__main__":
    main()
