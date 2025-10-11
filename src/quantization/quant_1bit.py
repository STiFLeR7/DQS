# src/quantization/quant_1bit.py
"""
Sign-based 1-bit quantizer for HuggingFace model checkpoints.
Saves per-parameter files under the output directory:
- For quantized weights (2D): <param_name>__signs.pt (int8), <param_name>__scale.pt (float32)
- For kept_fp params: <param_name>__fp16.pt (torch.float16)
- meta.json contains a list of quantized parameter names and kept params.

Usage:
    python src/quantization/quant_1bit.py \
      --base_dir "D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-base" \
      --out_dir  "D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-1bit" \
      --device cpu

Notes:
- Run on CPU to avoid using GPU memory. If you have ample RAM and want speed, use a CUDA device.
- This script is intentionally conservative: LayerNorm & bias kept in FP16.
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# ---- Helper utilities ----------------------------------------------------
def is_weight_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """
    Heuristic: quantize large 2D weight matrices (linear / qkv) and embeddings.
    Do NOT quantize LayerNorm / bias / small 1D tensors.
    """
    # shape-based: matrix (2D) with reasonable size
    if tensor.ndim == 2 and tensor.numel() >= 1024:
        # avoid LayerNorm and other small linear-like params that are 1D or tiny
        if ("layernorm" in name.lower()) or ("ln_" in name.lower()) or ("bias" in name.lower()):
            return False
        return True
    # embeddings are often 2D and should be quantized
    if tensor.ndim == 2 and tensor.numel() < 1024:
        # small matrices -> keep FP16 to be safe
        return False
    return False

def quantize_matrix_sign(t: torch.Tensor, per_row: bool = True):
    """
    Binarize a floating point matrix t into:
      signs: int8 tensor of {-1, +1} stored as int8 (values -1 or +1)
      scale: float32 per-row (or single scalar if per_row=False)
    Reconstruction: scale * signs (cast to float32/float16)
    """
    # convert to float32 for stable statistics
    t32 = t.to(torch.float32)
    if per_row and t32.ndim == 2:
        # scale per row (dim 1 columns)
        # use mean absolute value per row (L1) but add eps to avoid zeros
        eps = 1e-8
        scale = t32.abs().mean(dim=1, keepdim=True)  # shape (rows,1)
        signs = torch.sign(t32)
        # sign(0) -> 0; map zeros to +1
        signs[signs == 0] = 1.0
        signs_i8 = signs.to(torch.int8)
        return signs_i8, scale.to(torch.float32)
    else:
        # global scalar
        eps = 1e-8
        scale = t32.abs().mean()
        signs = torch.sign(t32)
        signs[signs == 0] = 1.0
        signs_i8 = signs.to(torch.int8)
        return signs_i8, torch.tensor(float(scale), dtype=torch.float32)

# ---- Main quantization routine ------------------------------------------
def quantize_checkpoint(base_dir: str, out_dir: str, device: str = "cpu"):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading config from {base_dir} (device={device})")
    # load config to ensure tokenizer/model class compatibility
    try:
        config = AutoConfig.from_pretrained(str(base_dir))
    except Exception:
        config = None

    # load model weights into CPU to iterate safely
    print("[+] Loading model (weights will be loaded to CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    state_dict = model.state_dict()
    print(f"[+] Loaded state_dict with {len(state_dict)} params")

    meta = {
        "base_dir": str(base_dir),
        "num_params": len(state_dict),
        "quantized": [],
        "kept_fp16": [],
    }

    # iterate and quantize param-by-param
    for name, tensor in tqdm(state_dict.items()):
        # skip optimizer / other keys that may appear (just in case)
        if not isinstance(tensor, torch.Tensor):
            continue

        # decide quantization
        q_ok = is_weight_quantizable(name, tensor)

        if q_ok:
            # quantize
            # move to CPU float32 for stable stats and operations
            t_cpu = tensor.detach().to(torch.float32).cpu()
            signs_i8, scale = quantize_matrix_sign(t_cpu, per_row=True)
            # save sign and scale tensors
            signs_path = out_dir / f"{name.replace('.', '__')}__signs.pt"
            scale_path = out_dir / f"{name.replace('.', '__')}__scale.pt"
            torch.save(signs_i8, signs_path)
            torch.save(scale, scale_path)
            meta["quantized"].append(name)
        else:
            # keep in fp16 to preserve model behavior for norm/bias/small tensors
            save_path = out_dir / f"{name.replace('.', '__')}__fp16.pt"
            torch.save(tensor.detach().to(torch.float16).cpu(), save_path)
            meta["kept_fp16"].append(name)

    # save metadata
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[+] Quantization complete.")
    print(f"    Quantized {len(meta['quantized'])} params, kept {len(meta['kept_fp16'])} params in fp16.")
    print(f"[+] Files written to: {out_dir}")
    return out_dir

# ---- Small loader for dequantization (for testing) -----------------------
def reconstruct_fp16_state_dict(quant_dir: str, out_state_path: str):
    """
    Reconstruct a full fp16 state_dict by dequantizing quantized params.
    Writes a single torch file at out_state_path containing the reconstructed state_dict.
    Use this to test inference or to convert back to a loadable HF checkpoint (not fully covered here).
    """

    quant_dir = Path(quant_dir)
    meta_path = quant_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found in quant_dir")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    state = {}
    # first load kept fp16 params
    for name in tqdm(meta["kept_fp16"], desc="loading kept fp16"):
        save_path = quant_dir / f"{name.replace('.', '__')}__fp16.pt"
        if not save_path.exists():
            raise FileNotFoundError(save_path)
        state[name] = torch.load(save_path).to(torch.float16)

    # now dequantize quantized params
    for name in tqdm(meta["quantized"], desc="reconstructing quantized"):
        signs_path = quant_dir / f"{name.replace('.', '__')}__signs.pt"
        scale_path = quant_dir / f"{name.replace('.', '__')}__scale.pt"
        if not signs_path.exists() or not scale_path.exists():
            raise FileNotFoundError(f"Missing quant artifacts for {name}")
        signs = torch.load(signs_path)           # int8
        scale = torch.load(scale_path)           # float32 (per-row or scalar)
        # convert signs to float16 and multiply by scale
        if isinstance(scale, torch.Tensor) and scale.ndim == 2 and signs.ndim == 2:
            # per-row scale (rows,1)
            reconstructed = signs.to(torch.float32) * scale
        else:
            # scalar
            reconstructed = signs.to(torch.float32) * float(scale)
        state[name] = reconstructed.to(torch.float16)

    # save single file
    torch.save(state, out_state_path)
    print("[+] Reconstructed state_dict saved to:", out_state_path)
    return out_state_path

# ---- CLI interface ------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="1-bit sign quantization (per-row scale) for HF models")
    p.add_argument("--base_dir", type=str, required=True, help="Path to base HF model directory")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for 1-bit artifacts")
    p.add_argument("--device", type=str, default="cpu", help="Device for loading (cpu or cuda)")
    p.add_argument("--reconstruct", action="store_true", help="Also reconstruct full fp16 state_dict for testing")
    p.add_argument("--recon_out", type=str, default=None, help="Output path for reconstructed state_dict (.pt)")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = quantize_checkpoint(args.base_dir, args.out_dir, device=args.device)
    if args.reconstruct:
        recon_out = args.recon_out or os.path.join(args.out_dir, "reconstructed_state_dict.pt")
        reconstruct_fp16_state_dict(out_dir, recon_out)

if __name__ == "__main__":
    main()
