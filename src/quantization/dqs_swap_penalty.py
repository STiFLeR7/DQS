# src/quantization/dqs_manager.py
"""
DQS Manager (FULL, updated)
- Fast, streaming 1-bit loader (per-param dequantize & in-place assign)
- 4-bit / 8-bit loading via BitsAndBytesConfig
- Proper tensor device movement (no BatchEncoding.to misuse)
- Caching and simple eviction policy (keep_loaded)
- Progress reporting for 1-bit loads

Usage:
    python src/quantization/dqs_manager.py

Edit LOCATORS and thresholds below to your setup.
"""
import time
import gc
import json
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from tqdm import tqdm

# ---------- USER PATHS (edit if necessary) ----------
MODEL_DIR = Path("D:/Phi3-EdgeQuant-Agent/models")

LOCATORS = {
    # 1-bit reconstructed artifacts folder (quant_1bit.py output)
    # expects MODEL_DIR / "Qwen2.5-3B-Instruct-1bit/" containing meta.json and per-param files
    "onebit_recon": MODEL_DIR / "Qwen2.5-3B-Instruct-1bit",
    "4bit": MODEL_DIR / "Qwen2.5-3B-Instruct-4bit-from-base",
    "8bit": MODEL_DIR / "Qwen2.5-3B-Instruct-8bit-from-base",
}

# token thresholds
def choose_label(n_tokens: int) -> str:
    if n_tokens <= 80:
        return "onebit_recon"
    if n_tokens <= 250:
        return "4bit"
    return "8bit"


class DQSManager:
    def __init__(self, locators: Dict[str, Path], keep_loaded: int = 1, onebit_device: str = "cpu"):
        self.locators = locators
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.keep_loaded = keep_loaded
        self.onebit_device = onebit_device  # "cpu" or "cuda"
        # cache base tokenizer once (for token counting)
        self.base_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"), use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    # ------------------ streaming 1-bit loader ------------------
    def _load_onebit_recon_streaming(self, quant_dir: Path, target_device: str = "cpu"):
        """
        Stream-loading of 1-bit quant artifacts into a freshly-instantiated model.

        Expects quant_dir produced by quant_1bit.py:
          - files named like <param.name.replace('.', '__')>__signs.pt
          - <param.name.replace('.', '__')>__scale.pt
          - kept params: <param.name.replace('.', '__')>__fp16.pt
          - meta.json listing 'quantized' and 'kept_fp16' names

        Returns (model, tokenizer). Model initially on CPU (or moved to CUDA if requested
        and available).
        """
        quant_dir = Path(quant_dir)
        meta_path = quant_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in quant dir: {quant_dir}")

        print(f"[1-bit] Streaming load from: {quant_dir}")
        # instantiate a model skeleton from base config (light)
        base_cfg = AutoConfig.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"))
        model = AutoModelForCausalLM.from_config(base_cfg)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"), use_fast=True)

        # utility to map param name -> file base
        def param_to_fname(name: str) -> str:
            return name.replace(".", "__")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        quantized = meta.get("quantized", [])
        kept = meta.get("kept_fp16", [])

        # build a name->parameter map (parameters + buffers)
        param_map = {k: v for k, v in model.named_parameters()}
        param_map.update({k: v for k, v in model.named_buffers()})

        # Load kept fp16 params first (usually small)
        print("[1-bit] Loading kept FP16 params...")
        for name in tqdm(kept, desc="kept_fp16"):
            fname = param_to_fname(name) + "__fp16.pt"
            ppath = quant_dir / fname
            if not ppath.exists():
                # warn and continue
                tqdm.write(f"[1-bit] Warning: missing kept file {ppath}; skipping")
                continue
            tensor = torch.load(str(ppath), map_location="cpu")
            if name in param_map:
                with torch.no_grad():
                    tgt = param_map[name].data
                    # safe copy (cast to target dtype)
                    try:
                        tgt.copy_(tensor.to(tgt.dtype))
                    except Exception as e:
                        tqdm.write(f"[1-bit] Warning copying kept param {name}: {e}")
            else:
                tqdm.write(f"[1-bit] Warning: param {name} not present in model; skipping")

        # Now stream quantized params
        print("[1-bit] Loading quantized params (streaming dequantize)...")
        for name in tqdm(quantized, desc="quantized"):
            signs_file = quant_dir / (param_to_fname(name) + "__signs.pt")
            scale_file = quant_dir / (param_to_fname(name) + "__scale.pt")
            if not signs_file.exists() or not scale_file.exists():
                tqdm.write(f"[1-bit] Warning: missing quant artifacts for {name}; skipping")
                continue

            signs = torch.load(str(signs_file), map_location="cpu")   # int8 (-1/+1)
            scale = torch.load(str(scale_file), map_location="cpu")   # float32 per-row or scalar

            # reconstruct
            if isinstance(scale, torch.Tensor) and scale.ndim == 2 and signs.ndim == 2:
                recon = signs.to(torch.float32) * scale
            else:
                recon = signs.to(torch.float32) * float(scale)

            # assign if param exists
            if name in param_map:
                with torch.no_grad():
                    tgt = param_map[name].data
                    if recon.shape != tgt.shape:
                        # try transpose heuristic
                        if recon.T.shape == tgt.shape:
                            recon = recon.T
                        else:
                            # try reshape if numel matches
                            if recon.numel() == tgt.numel():
                                try:
                                    recon = recon.view(tgt.shape)
                                except Exception:
                                    tqdm.write(f"[1-bit] Skipping {name}: cannot reshape {recon.shape} to {tgt.shape}")
                                    continue
                            else:
                                tqdm.write(f"[1-bit] Skipping {name}: shape mismatch recon={recon.shape} target={tgt.shape}")
                                continue
                    try:
                        tgt.copy_(recon.to(tgt.dtype))
                    except Exception as e:
                        tqdm.write(f"[1-bit] Warning copying quant param {name}: {e}")
            else:
                tqdm.write(f"[1-bit] Warning: quant param {name} not found in model; skipping")

        # move to device if requested
        if target_device == "cuda" and torch.cuda.is_available():
            try:
                model.to("cuda")
            except Exception as e:
                print("[1-bit] Warning: move to CUDA failed:", e)

        print("[1-bit] streaming load complete.")
        return model, tokenizer

    # ------------------ bitsandbytes loader ------------------
    def _load_bitsandbytes_model(self, path: Path):
        """Load a folder saved by the quantization script using BitsAndBytesConfig."""
        lower = path.name.lower()
        if "4bit" in lower:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        return model, tokenizer

    # ------------------ public load / generate ------------------
    def load_label(self, label: str):
        """Return (entry, cold_load_bool). Entry contains keys 'model' and 'tokenizer'."""
        if label in self.cache:
            return self.cache[label], False

        loc = self.locators[label]
        print(f"[DQS] Loading label={label} from {loc}")
        if label == "onebit_recon":
            # streaming 1-bit load (target device set to CPU by default)
            model, tokenizer = self._load_onebit_recon_streaming(loc, target_device=self.onebit_device)
        else:
            model, tokenizer = self._load_bitsandbytes_model(loc)

        try:
            model.eval()
        except Exception:
            pass

        self.cache[label] = {"model": model, "tokenizer": tokenizer}
        # prune cache
        self._prune_cache(keep_label=label)
        return self.cache[label], True

    def _prune_cache(self, keep_label: str):
        # keep at most self.keep_loaded labels; try to retain keep_label
        keys = [k for k in list(self.cache.keys()) if k != keep_label]
        while len(self.cache) > self.keep_loaded:
            victim = keys.pop(0)
            print(f"[DQS] Unloading cached model: {victim}")
            try:
                del self.cache[victim]
            except Exception:
                pass
            self.clear_cuda()

    def generate(self, prompt: str, max_new_tokens: int = 64):
        # count tokens with base tokenizer
        n = len(self.base_tokenizer(prompt)["input_ids"])
        label = choose_label(n)
        (entry, cold) = self.load_label(label)
        model = entry["model"]
        tokenizer = entry["tokenizer"]

        # determine model device (fall back to manager device)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = self.device

        # move model to GPU if available and currently on CPU
        if torch.cuda.is_available() and str(model_device).startswith("cpu"):
            try:
                model.to("cuda")
                model_device = next(model.parameters()).device
            except Exception:
                pass

        # tokenize and move inputs explicitly to model device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # measure generation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak = float(torch.cuda.max_memory_allocated()) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return {"label": label, "cold_load": cold, "gen_time_s": dt, "gpu_peak_gb": peak, "text": text}


# ------------------ quick example usage ------------------
if __name__ == "__main__":
    mgr = DQSManager(LOCATORS, keep_loaded=1, onebit_device="cpu")
    prompts = [
        "Short query about photosynthesis?",
        "Write a detailed review of the 2021 IPCC report and summarize mitigation strategies across agriculture, energy, and transport in ~200 words."
    ]
    for p in prompts:
        print("\n-- prompt:", p)
        t_start = time.time()
        res = mgr.generate(p, max_new_tokens=80)
        duration = time.time() - t_start
        print(f"Result (elapsed {duration:.3f}s): {res}")
