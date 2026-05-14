"""
Download Fish Audio S2-Pro model weights from HuggingFace.

Run once before starting the server:
  uv run python download_models.py

Downloads the open-source S2-Pro weights to ./checkpoints/s2-pro/.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "fishaudio/s2-pro"
LOCAL_DIR = Path("checkpoints/s2-pro")

REQUIRED_FILES = [
    "codec.pth",
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def models_present() -> bool:
    return all((LOCAL_DIR / f).exists() for f in REQUIRED_FILES)


def download() -> None:
    if models_present():
        print(f"Models already present at {LOCAL_DIR}. Nothing to download.")
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to {LOCAL_DIR} ...")
    print("This may take several minutes depending on your connection.\n")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.git*", "*.gitattributes", "*.md"],
    )

    if models_present():
        print(f"\nDownload complete. Files in {LOCAL_DIR}:")
        for f in sorted(LOCAL_DIR.iterdir()):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:55s} {size_mb:7.1f} MB")
    else:
        missing = [f for f in REQUIRED_FILES if not (LOCAL_DIR / f).exists()]
        raise RuntimeError(f"Download seems incomplete. Missing: {missing}")


if __name__ == "__main__":
    download()
