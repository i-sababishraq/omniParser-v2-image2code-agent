# Environment Setup and Running the Pipeline

This guide explains how to activate the environment and run the OmniParser v2 sample pipeline on this machine.

## 1) Prerequisites
- Conda environment already present at:
  - `/anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni`
- Project root:
  - `/anvil/projects/x-soc250046/x-sishraq/omniparser-v2`
- Weights:
  - YOLO icon detector: `weights/icon_detect/model.pt`
  - Florence-2 local folder: `weights/icon_caption_florence/`

## 2) Activate environment
```bash
source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
conda activate /anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni
```

## 3) Install Python dependencies (one-time)
We pin transformers to a Florence-compatible version and use a small local `flash_attn` stub so the Florence modeling code can import.

```bash
# from repo root
pip install -U --no-input -r requirements.txt
```

Notes:
- `requirements.txt` pins `transformers==4.41.1`.
- A minimal `flash_attn` stub lives in `flash_attn/__init__.py`.
  - If you want real FlashAttention, install `flash-attn` that matches your CUDA and PyTorch versions instead of the stub.

## 4) Verify versions
(Optional, but helpful when debugging.)
```bash
python - <<'PY'
import torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
PY
```

## 5) Run the 5-sample pipeline
The script processes 5 images per split (android/ios/web), saving annotated images and JSON.

```bash
# from repo root
export PYTHONPATH=$(pwd)
python scripts/run_samples.py
```

If you prefer to be explicit with the env’s Python:
```bash
PYTHONPATH=$(pwd) /anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni/bin/python scripts/run_samples.py
```

## 6) Outputs
- Annotated images: `results_samples/<split>/annotated/*.png`
- JSON metadata: `results_samples/<split>/json/*.json`
- Summary: `results_samples/summary.json`

## 7) Troubleshooting
- If you see an error about `flash_attn` missing, ensure you’re using this repo version (which includes the stub). For real acceleration, install a compatible `flash-attn` build.
- If you see PyTorch/CUDA errors, confirm the conda env is active and your GPU is available.
- OCR backends:
  - PaddleOCR is enabled by default in `scripts/run_samples.py`.
  - You can toggle to EasyOCR in `util/utils.py:check_ocr_box` if needed.

## 8) Re-running
You can re-run the pipeline any time with the same commands in step 5. Existing outputs will be overwritten.
