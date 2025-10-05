#!/bin/bash

# Usage: ./run_inference.sh /path/to/image.png
# If no image is provided, the script will run the sample batch runner.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"

# Activate conda env and exports so this script can be run directly with the provided launcher
source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh || true
conda activate /anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni || true
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local

# Conda env python (now available after activation)
CONDA_PYTHON="/anvil/projects/x-soc250046/x-sishraq/.conda/envs/omni/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
	echo "Warning: expected Python not found at $CONDA_PYTHON"
	echo "Falling back to system python"
	CONDA_PYTHON="$(which python)"
fi

echo "Using Python: $CONDA_PYTHON"
$CONDA_PYTHON --version || true
echo "PYTHONPATH: $PYTHONPATH"

if [ "$#" -eq 0 ]; then
	echo "No image provided — running default sample runner"
	exec "$CONDA_PYTHON" "$ROOT/scripts/run_samples.py"
fi

# Use the provided image path
IMG_PATH="$1"
# Optionally resolve to absolute path if available
if command -v readlink >/dev/null 2>&1; then
	IMG_PATH="$(readlink -f "$IMG_PATH" || echo "$IMG_PATH")"
fi
if [ ! -f "$IMG_PATH" ]; then
	echo "Image not found: $IMG_PATH"
	exit 2
fi

echo "Running single-image inference for: $IMG_PATH"

# Use a small inline Python wrapper to reuse existing functions and keep outputs under results_samples/single/
ROOT="$ROOT" IMG_PATH="$IMG_PATH" "$CONDA_PYTHON" - <<'PY'
import json, sys, os
from pathlib import Path
ROOT = Path(os.environ.get('ROOT', '.'))
sys.path.insert(0, str(ROOT))
from PIL import Image
from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

WEIGHTS = ROOT / 'weights'
som_model_path = WEIGHTS / 'icon_detect' / 'model.pt'
caption_path = WEIGHTS / 'icon_caption_florence'
if not som_model_path.exists() or not caption_path.exists():
		print('ERROR: missing weights. See SETUP_ENV.md to download model weights.')
		sys.exit(1)

som_model = get_yolo_model(str(som_model_path))
caption_processor = get_caption_model_processor(model_name='florence2', model_name_or_path=str(caption_path))

img_path = os.environ.get('IMG_PATH')
if not img_path or not os.path.isfile(img_path):
	print(f'ERROR: invalid image path passed: {img_path!r}')
	sys.exit(2)
img = Image.open(img_path)
ocr_res, _ = check_ocr_box(img, display_img=False, output_bb_format='xyxy', use_paddleocr=True)
ocr_text, ocr_bb = ocr_res
encoded_img, label_coords, parsed_boxes = get_som_labeled_img(img, model=som_model, BOX_TRESHOLD=0.01, output_coord_in_ratio=True, ocr_bbox=ocr_bb, draw_bbox_config=None, caption_model_processor=caption_processor, ocr_text=ocr_text, use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

out_dir = ROOT / 'results_samples' / 'single'
out_dir.mkdir(parents=True, exist_ok=True)
in_path = Path(img_path)
out_png = out_dir / (in_path.stem + '_annotated.png')
out_json = out_dir / (in_path.stem + '.json')
with open(out_png, 'wb') as f:
		f.write(__import__('base64').b64decode(encoded_img))
payload = {'image': str(in_path), 'ocr_text': ocr_text, 'label_coordinates': label_coords, 'parsed_boxes': parsed_boxes}
out_json.write_text(json.dumps(payload, indent=2))
print('Wrote:', out_png, out_json)
PY

echo "Done. Outputs in results_samples/single/"

