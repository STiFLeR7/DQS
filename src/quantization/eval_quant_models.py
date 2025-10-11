# src/quantization/eval_quant_models.py
"""
Robust evaluator for quantized models (Phi3-EdgeQuant-Agent).
Measures:
 - model size (on-disk)
 - load time
 - generation latency
 - GPU peak bytes (torch.cuda.max_memory_allocated)
 - RSS memory before/after (psutil if available)

Usage:
    python src/quantization/eval_quant_models.py            # run all models under BASE
    python src/quantization/eval_quant_models.py --limit 3  # test only first 3
"""

import os, time, json, sys, argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# User-config (edit if needed)
BASE = Path("D:/Phi3-EdgeQuant-Agent/models/quantized")
OUT_DIR = Path.cwd() / "outputs" / "quant_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT = "Explain knowledge distillation in AI in simple terms."
MAX_NEW_TOKENS = 120

# Utilities
def human_size(path: Path):
    try:
        total = 0
        if path.is_file():
            total = path.stat().st_size
        else:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        for unit in ("B","KB","MB","GB","TB"):
            if total < 1024.0:
                return f"{total:3.2f}{unit}"
            total /= 1024.0
        return f"{total:.2f}PB"
    except Exception:
        return "N/A"

def get_rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return None

def write_output_txt(name, text, meta):
    p = OUT_DIR / f"{name}_output.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# model: {name}\n# time: {datetime.utcnow().isoformat()}Z\n\n")
        f.write(text or "")
    return str(p)

def safe_reset_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

# --- Loader attempts ---

def try_autogptq_load_generate(model_path: Path, prompt: str):
    """
    Try using auto_gptq (AutoGPTQForCausalLM). Returns ((text, meta), None) on success,
    or (None, error_str) on failure.
    """
    try:
        import torch
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as e:
        return None, f"auto-gptq-missing: {e}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False, trust_remote_code=True)
        # AutoGPTQ loader - try from_quantized then from_pretrained
        try:
            model = AutoGPTQForCausalLM.from_quantized(str(model_path), device=device, use_safetensors=True, trust_remote_code=True)
            loader_name = "AutoGPTQ.from_quantized"
        except Exception:
            model = AutoGPTQForCausalLM.from_pretrained(str(model_path), device=device, use_safetensors=True, trust_remote_code=True)
            loader_name = "AutoGPTQ.from_pretrained"
        load_time = time.perf_counter() - start

        # generate
        safe_reset_cuda()
        gen_start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        if device.startswith("cuda"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.to(device)
        gen_out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        gen_time = time.perf_counter() - gen_start

        # GPU peak
        gpu_peak = 0
        try:
            gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except Exception:
            gpu_peak = 0

        text = tokenizer.decode(gen_out[0], skip_special_tokens=True) if hasattr(tokenizer, "decode") else str(gen_out)
        meta = {"loader": loader_name, "load_time_s": load_time, "gen_time_s": gen_time, "gpu_peak_bytes": gpu_peak}
        return (text, meta), None
    except Exception as e:
        return None, f"auto-gptq-error: {e}"

def try_gptqmodel_load_generate(model_path: Path, prompt: str):
    """
    Try older gptqmodel (gptqmodel.GPTQModel) loader if installed.
    """
    try:
        import torch
        from transformers import AutoTokenizer
        from gptqmodel import GPTQModel
    except Exception as e:
        return None, f"gptqmodel-missing: {e}"

    try:
        start = time.perf_counter()
        # Many GPTQModel implementations expose from_pretrained/from_quantized
        try:
            model = GPTQModel.from_pretrained(str(model_path), device="cuda")
            loader_name = "GPTQModel.from_pretrained"
        except Exception:
            if hasattr(GPTQModel, "from_quantized"):
                model = GPTQModel.from_quantized(str(model_path), device="cuda")
                loader_name = "GPTQModel.from_quantized"
            else:
                model = GPTQModel.from_pretrained(str(model_path), device="cuda")
                loader_name = "GPTQModel.from_pretrained"
        load_time = time.perf_counter() - start

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        safe_reset_cuda()
        gen_start = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        gen_time = time.perf_counter() - gen_start
        gpu_peak = 0
        try:
            gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except Exception:
            gpu_peak = 0

        if isinstance(out, str):
            text = out
        else:
            text = tokenizer.decode(out[0], skip_special_tokens=True) if tokenizer is not None else str(out)

        meta = {"loader": loader_name, "load_time_s": load_time, "gen_time_s": gen_time, "gpu_peak_bytes": gpu_peak}
        return (text, meta), None
    except Exception as e:
        return None, f"gptqmodel-error: {e}"

def try_transformers_bnb_load_generate(model_path: Path, prompt: str):
    """
    Use transformers + BitsAndBytesConfig for bnb loading.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            # prefer official BitsAndBytesConfig
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        except Exception:
            # fallback minimal config shim
            class BitsAndBytesConfig:
                def __init__(self, **kw):
                    self.__dict__.update(kw)
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
    except Exception as e:
        return None, f"transformers-bnb-missing: {e}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        return None, f"tokenizer-error: {e}"

    try:
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(str(model_path),
                                                    quantization_config=bnb_cfg,
                                                    device_map="auto",
                                                    trust_remote_code=True)
        load_time = time.perf_counter() - start
    except Exception as e:
        return None, f"transformers-bnb-load-error: {e}"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        safe_reset_cuda()
        gen_start = time.perf_counter()
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        gen_time = time.perf_counter() - gen_start
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gpu_peak = 0
        try:
            gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except Exception:
            gpu_peak = 0
        meta = {"loader": "transformers+bitsandbytes", "load_time_s": load_time, "gen_time_s": gen_time, "gpu_peak_bytes": gpu_peak}
        return (decoded, meta), None
    except Exception as e:
        return None, f"transformers-bnb-gen-error: {e}"

def try_transformers_plain_load_generate(model_path: Path, prompt: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        return None, f"transformers-missing: {e}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        return None, f"tokenizer-error: {e}"

    try:
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(str(model_path), device_map="auto", trust_remote_code=True)
        load_time = time.perf_counter() - start
    except Exception as e:
        return None, f"transformers-load-error: {e}"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        safe_reset_cuda()
        gen_start = time.perf_counter()
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        gen_time = time.perf_counter() - gen_start
        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        gpu_peak = 0
        try:
            gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except Exception:
            gpu_peak = 0
        meta = {"loader": "transformers_plain", "load_time_s": load_time, "gen_time_s": gen_time, "gpu_peak_bytes": gpu_peak}
        return (decoded, meta), None
    except Exception as e:
        return None, f"transformers-gen-error: {e}"

# ---- Main loop ----
def main(limit: int = None):
    results = []
    if not BASE.exists():
        print("BASE not found:", BASE)
        sys.exit(1)

    candidates = sorted([p for p in BASE.iterdir() if p.is_dir()])
    if limit:
        candidates = candidates[:limit]

    for model_path in tqdm(candidates, desc="models"):
        print("\n---", model_path.name, "---")
        rec = {"model_name": model_path.name, "path": str(model_path), "timestamp": datetime.utcnow().isoformat() + "Z"}
        rec["exists"] = model_path.exists()
        rec["size"] = human_size(model_path) if model_path.exists() else "N/A"

        files = [f.name for f in model_path.iterdir() if f.is_file()]

        # Quick skip heuristic (if needed)
        if not files:
            rec["skipped"] = True
            rec["note"] = "empty-folder"
            print("Skipping empty folder")
            results.append(rec)
            continue

        rss_before = get_rss_bytes()

        # Try loaders in order. Prefer AutoGPTQ -> gptqmodel -> bnb -> plain transformers
        out_text = None
        meta = {}
        error = None

        # 1) AutoGPTQ
        res, err = try_autogptq_load_generate(model_path, PROMPT)
        if res:
            out_text, meta = res
        else:
            # 2) gptqmodel
            res2, err2 = try_gptqmodel_load_generate(model_path, PROMPT)
            if res2:
                out_text, meta = res2
                err = None
            else:
                # 3) transformers + bitsandbytes
                res3, err3 = try_transformers_bnb_load_generate(model_path, PROMPT)
                if res3:
                    out_text, meta = res3
                    err = None
                else:
                    # 4) plain transformers
                    res4, err4 = try_transformers_plain_load_generate(model_path, PROMPT)
                    if res4:
                        out_text, meta = res4
                        err = None
                    else:
                        # collect first non-empty error for reporting
                        err = err or err2 or err3 or err4

        rss_after = get_rss_bytes()
        rec.update({
            "loader": meta.get("loader") if meta else None,
            "load_time_s": meta.get("load_time_s") if meta else None,
            "gen_time_s": meta.get("gen_time_s") if meta else None,
            "gpu_peak_bytes": meta.get("gpu_peak_bytes") if meta else None,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "success": out_text is not None,
            "error": err,
        })

        if out_text:
            txt_path = write_output_txt(model_path.name, out_text, rec)
            rec["output_file"] = txt_path
            print(f"OK: loader={rec['loader']}, load_s={rec['load_time_s']:.3f}, gen_s={rec['gen_time_s']:.3f}, gpu_peak={rec['gpu_peak_bytes']}")
        else:
            print("FAIL:", rec["error"])

        results.append(rec)

    # write compact JSON
    out_json = OUT_DIR / "benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nWrote", out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of models to evaluate")
    args = parser.parse_args()
    main(limit=args.limit)
