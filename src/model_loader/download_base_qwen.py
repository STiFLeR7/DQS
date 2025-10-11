# download_base_qwen.py
from huggingface_hub import snapshot_download
from pathlib import Path

BASE = Path(r"D:\Phi3-EdgeQuant-Agent\models")
BASE.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir=str(BASE / "Qwen2.5-3B-Instruct-base"),
    allow_patterns=["*"],
)

print("✅ Base model downloaded to D:/Phi3-EdgeQuant-Agent/models/Qwen2.5-3B-Instruct-base")
