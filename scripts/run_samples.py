#!/usr/bin/env python3
"""
Quick runner: run OmniParser V2 pipeline on 5 samples from each split (android/ios/web).
Saves annotated images and JSON outputs into results/<split>/.
"""
import os
import sys
import base64
import json
from pathlib import Path
from PIL import Image
import traceback

from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
RESULTS = ROOT / "results_samples"

SPLITS = [
    (DATA / "android" / "all_data", "android"),
    (DATA / "ios" / "all_data", "ios"),
    (DATA / "web" / "all_data", "web"),
]

NUM_SAMPLES = 5
BOX_TRESHOLD = 0.01


def pick_samples(folder: Path, n: int):
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    return files[:n]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_base64_png(b64str: str, outpath: Path):
    data = base64.b64decode(b64str)
    outpath.write_bytes(data)


def main():
    print('Running samples runner from', ROOT)

    som_model_path = WEIGHTS / 'icon_detect' / 'model.pt'
    caption_path = WEIGHTS / 'icon_caption_florence'

    if not som_model_path.exists():
        print('ERROR: SOM model not found at', som_model_path)
        sys.exit(1)
    if not caption_path.exists():
        print('ERROR: caption model folder not found at', caption_path)
        sys.exit(1)

    print('Loading SOM (YOLO) model...')
    som_model = get_yolo_model(str(som_model_path))
    print('Loading caption model (Florence) ...')
    caption_processor = get_caption_model_processor(model_name='florence2', model_name_or_path=str(caption_path))

    summary = {}

    for folder, name in SPLITS:
        print('\nProcessing split:', name, 'from', folder)
        out_dir = RESULTS / name
        annotated_dir = out_dir / 'annotated'
        json_dir = out_dir / 'json'
        ensure_dir(annotated_dir)
        ensure_dir(json_dir)

        samples = pick_samples(folder, NUM_SAMPLES)
        print(f'Found {len(samples)} images, using {len(samples[:NUM_SAMPLES])} samples')
        summary[name] = []

        for img_path in samples:
            print('->', img_path.name)
            try:
                # Load image first
                image = Image.open(img_path)
                print(f'  Image size: {image.size}, mode: {image.mode}')
                
                # OCR first
                (ocr_text, ocr_bb), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold':0.8}, use_paddleocr=True)
                print(f'  OCR found {len(ocr_text)} text elements')

                # Run main pipeline
                encoded_img, label_coords, parsed_boxes = get_som_labeled_img(image, model=som_model, BOX_TRESHOLD=BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bb, draw_bbox_config=None, caption_model_processor=caption_processor, ocr_text=ocr_text, use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

                # Save annotated image (base64 PNG)
                annotated_out = annotated_dir / (img_path.stem + '_annotated.png')
                save_base64_png(encoded_img, annotated_out)

                # Save JSON result
                json_out = json_dir / (img_path.stem + '.json')
                payload = {
                    'image': str(img_path),
                    'ocr_text': ocr_text,
                    'label_coordinates': label_coords,
                    'parsed_boxes': parsed_boxes,
                }
                json_out.write_text(json.dumps(payload, indent=2))

                summary[name].append({
                    'image': str(img_path.name),
                    'annotated': str(annotated_out.relative_to(ROOT)),
                    'json': str(json_out.relative_to(ROOT)),
                    'num_boxes': len(parsed_boxes)
                })

            except Exception as e:
                print('Error processing', img_path, e)
                print(traceback.format_exc())
                summary[name].append({'image': str(img_path.name), 'error': str(e)})

    # Save overall summary
    ensure_dir(RESULTS)
    summary_path = RESULTS / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print('\nDone. Results written to', RESULTS)
    print('Summary saved to', summary_path)


if __name__ == '__main__':
    main()
