# download_qwen_models.py
from huggingface_hub import snapshot_download
from pathlib import Path

BASE = Path(r"D:\Phi3-EdgeQuant-Agent\models")
BASE.mkdir(parents=True, exist_ok=True)

repos = {
    "4bit": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "8bit": "mlx-community/Qwen2.5-3B-Instruct-8bit",
}

for k, repo in repos.items():
    target = BASE / f"Qwen2.5-3B-Instruct-{k}"
    print("Downloading", repo, "->", target)
    snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=["*"])
    print("Finished", repo)
print("All downloads completed.")
