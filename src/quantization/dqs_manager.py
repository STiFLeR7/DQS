# src/quantization/dqs_manager.py
"""
DQS Manager (fixed)
- Token-length based model selection (1-bit recon / 4-bit / 8-bit)
- Lazy loads models, caches them, moves inputs correctly to model device
- Measures cold vs warm loads and per-request timings

Edit LOCATORS and thresholds below to your liking.
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

MODEL_DIR = Path("D:/Phi3-EdgeQuant-Agent/models")

LOCATORS = {
    # if you have a reconstructed state_dict for 1-bit:
    "onebit_recon": MODEL_DIR / "Qwen2.5-3B-Instruct-1bit" / "reconstructed_state_dict.pt",
    "4bit": MODEL_DIR / "Qwen2.5-3B-Instruct-4bit-from-base",
    "8bit": MODEL_DIR / "Qwen2.5-3B-Instruct-8bit-from-base",
}

# thresholds (tokens)
def choose_label(n_tokens: int) -> str:
    if n_tokens <= 80:
        return "onebit_recon"
    if n_tokens <= 250:
        return "4bit"
    return "8bit"

# DQS manager
class DQSManager:
    def __init__(self, locators: Dict[str, Path], keep_loaded: int = 1):
        self.locators = locators
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.keep_loaded = keep_loaded
        # cache base tokenizer once (for token counting)
        self.base_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"), use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clear_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def _load_onebit_recon(self, path: Path):
        """
        Load the reconstructed state_dict into a model instance.
        Expects path -> reconstructed_state_dict.pt (a state_dict saved via torch.save)
        """
        if not path.exists():
            raise FileNotFoundError(f"1-bit reconstructed file not found: {path}")

        # Load base config and instantiate model skeleton
        config = AutoConfig.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"))
        model = AutoModelForCausalLM.from_config(config)
        state = torch.load(str(path), map_location="cpu")
        # load into model (non-strict to allow small mismatches)
        model.load_state_dict(state, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "Qwen2.5-3B-Instruct-base"), use_fast=True)
        return model, tokenizer

    def _load_bitsandbytes_model(self, path: Path):
        """Load a folder saved by quantize_bnb_and_benchmark.py using BitsAndBytesConfig."""
        # detect 4bit or 8bit by folder name
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

    def load_label(self, label: str):
        """Return (entry, cold_load_bool). Entry contains keys 'model' and 'tokenizer'."""
        if label in self.cache:
            return self.cache[label], False

        loc = self.locators[label]
        print(f"[DQS] Loading label={label} from {loc}")
        if label == "onebit_recon":
            model, tokenizer = self._load_onebit_recon(loc)
            # move model to cpu initially; we'll move to GPU on generate if needed
        else:
            model, tokenizer = self._load_bitsandbytes_model(loc)

        try:
            model.eval()
        except Exception:
            pass

        self.cache[label] = {"model": model, "tokenizer": tokenizer}
        # enforce cache size
        self._prune_cache(keep_label=label)
        return self.cache[label], True

    def _prune_cache(self, keep_label: str):
        # keep at most self.keep_loaded labels; try to keep 'keep_label'
        keys = list(self.cache.keys())
        # ensure keep_label present first
        keys = [k for k in keys if k != keep_label]
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

        # ensure model device and move input tensors properly
        model_device = None
        # if model has parameters, get a device from them; otherwise use self.device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = self.device

        # move model to GPU if available and not already on GPU
        if torch.cuda.is_available() and str(model_device).startswith("cpu"):
            try:
                model.to("cuda")
                model_device = next(model.parameters()).device
            except Exception:
                pass

        # tokenize and move tensors explicitly
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

# Example usage
if __name__ == "__main__":
    mgr = DQSManager(LOCATORS, keep_loaded=1)
    prompts = [
        "Short query about photosynthesis?",
        "Write a detailed review of the 2021 IPCC report and summarize mitigation strategies across agriculture, energy, and transport in ~200 words."
    ]
    for p in prompts:
        print("\n-- prompt:", p)
        res = mgr.generate(p, max_new_tokens=80)
        print(res)
