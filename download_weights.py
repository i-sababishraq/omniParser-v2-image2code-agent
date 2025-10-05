from huggingface_hub import hf_hub_download
import os

# Create weights directory
os.makedirs("weights", exist_ok=True)

# Files to download
files = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors"
]

print("Downloading OmniParser v2 model weights...")
for file in files:
    print(f"Downloading {file}...")
    hf_hub_download(
        repo_id="microsoft/OmniParser-v2.0",
        filename=file,
        local_dir="weights",
        local_dir_use_symlinks=False
    )
    print(f"✓ Downloaded {file}")

# Rename icon_caption to icon_caption_florence
if os.path.exists("weights/icon_caption"):
    if os.path.exists("weights/icon_caption_florence"):
        print("Removing existing icon_caption_florence directory...")
        import shutil
        shutil.rmtree("weights/icon_caption_florence")
    os.rename("weights/icon_caption", "weights/icon_caption_florence")
    print("✓ Renamed icon_caption to icon_caption_florence")

print("\n✓ All model weights downloaded successfully!")
