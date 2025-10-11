# download_hf_repo.py
"""Download HF model repo to a local folder without loading into memory.

Usage:
    python -m src.quantization.download_hf_repo
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
import sys

# CHANGE THESE if you want
HF_REPO_ID = "kaitchup/Phi-3-mini-4k-instruct-gptq-4bit"
LOCAL_DIR = Path("D:/Phi3-EdgeQuant-Agent/models/quantized/Phi-3-mini-4k-instruct-4bit-gptq")

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔽 snapshot_download('{HF_REPO_ID}') -> {LOCAL_DIR}")
    try:
        # snapshot_download returns the path to the HF cache folder for that repo.
        cache_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=None, ignore_patterns=None)
        print("✅ Downloaded to HF cache:", cache_path)

        # Copy everything from the repo cache folder into LOCAL_DIR (overwrites files with same name)
        # The snapshot_download returns a path like: <cache>/datasets/... or <cache>/repos/<user>/<repo>/...
        # We assume it is the correct repo root containing model files.
        from os import listdir
        from os.path import join, isdir

        # copy file-by-file to avoid moving unrelated files
        import filecmp
        for item in Path(cache_path).iterdir():
            dest = LOCAL_DIR / item.name
            if item.is_dir():
                # copytree with dirs_exist_ok
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"✅ All repo files copied to: {LOCAL_DIR}")
        print("Files present:", [p.name for p in LOCAL_DIR.iterdir()][:40])
        print("\nYou can now load the model from this folder later with `AutoModelForCausalLM.from_pretrained(LOCAL_DIR, trust_remote_code=True)` on a machine with proper native libs.")
    except Exception as e:
        print("❗ Failed to download/copy repo files:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
