#!/usr/bin/env python3
"""
Validate Layout Detection - Comprehensive validation of VLM layout understanding.

This script:
1. Shows the source of coordinates (OmniParser bounding boxes)
2. Runs coordinate-based analysis (algorithmic approach)
3. Runs VLM-based analysis (AI understanding)
4. Compares both approaches for horizontal AND vertical alignment
5. Generates visual validation with annotated images
6. Produces accuracy metrics

Usage:
    python validate_layout_detection.py --input results_samples/single/demo_image_enhanced.json \
                                        --image imgs/demo_image.jpg \
                                        --output-dir results_samples/validation
"""

import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Import our layout detection modules
from util.layout_detector import (
    detect_horizontal_rows,
    detect_vertical_columns,
    detect_containers,
    analyze_layout
)

# Try to import PIL for visualization
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, visual validation will be skipped")

# Try to import VLM modules
try:
    import requests
    import base64
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("Warning: requests not available, VLM validation will be skipped")


def load_data(json_path: str) -> Tuple[List[Dict], Dict, str]:
    """Load enhanced JSON data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    boxes = data['parsed_content_list']
    dimensions = data['image_dimensions']
    image_path = data.get('image', '')
    
    print(f"\n{'='*80}")
    print(f"DATA SOURCE: OmniParser Enhanced JSON")
    print(f"{'='*80}")
    print(f"File: {json_path}")
    print(f"Total boxes detected: {len(boxes)}")
    print(f"Image dimensions: {dimensions['width']}x{dimensions['height']}px")
    print(f"Image path: {image_path}")
    print(f"\n[DATA] Coordinate Structure (from OmniParser):")
    print(f"- Normalized coordinates (0-1 range)")
    print(f"- Pixel coordinates (absolute)")
    print(f"- Center point (x, y)")
    print(f"- Size (width, height, area)")

    # Show example box
    if boxes:
        print(f"\n Example Box (Box 0):")
        box = boxes[0]
        print(f"Content: '{box.get('content', 'N/A')}'")
        print(f"Type: {box.get('type', 'unknown')}")
        print(f"Coordinates:")
        print(f"- Center: ({box['coordinates']['center']['x']}, {box['coordinates']['center']['y']})")
        print(f"- Pixels: ({box['coordinates']['pixels']['x1']}, {box['coordinates']['pixels']['y1']}) to ({box['coordinates']['pixels']['x2']}, {box['coordinates']['pixels']['y2']})")
        print(f"- Size: {box['coordinates']['size']['width']}x{box['coordinates']['size']['height']}px")

    return boxes, dimensions, image_path


def run_coordinate_analysis(boxes: List[Dict], y_threshold: int = 10, x_threshold: int = 10) -> Dict:
    """
    Run algorithmic coordinate-based analysis.
    
    Args:
        boxes: List of bounding boxes
        y_threshold: Y-axis threshold for horizontal alignment (pixels)
        x_threshold: X-axis threshold for vertical alignment (pixels)
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n{'='*80}")
    print(f"COORDINATE-BASED ANALYSIS (Algorithmic Approach)")
    print(f"{'='*80}")
    print(f"Using thresholds: Y±{y_threshold}px (horizontal), X±{x_threshold}px (vertical)")
    
    # Detect horizontal rows
    print(f"\n[DETECT] Detecting horizontal rows...")
    rows = detect_horizontal_rows(boxes, y_threshold=y_threshold)
    print(f"[OK] Found {len(rows)} horizontal rows")
    
    # Show top 5 rows
    for i, row in enumerate(rows[:5]):
        avg_y = sum(b['coordinates']['center']['y'] for b in row) / len(row)
        contents = [b.get('content', 'N/A')[:20] for b in row[:3]]
        print(f"Row {i+1} (Y≈{avg_y:.1f}): {len(row)} boxes - {', '.join(contents)}...")
    
    # Detect vertical columns
    print(f"\n[DETECT] Detecting vertical columns...")
    columns = detect_vertical_columns(boxes, x_threshold=x_threshold)
    print(f"[OK] Found {len(columns)} vertical columns")
    
    # Show top 5 columns
    for i, col in enumerate(columns[:5]):
        avg_x = sum(b['coordinates']['center']['x'] for b in col) / len(col)
        contents = [b.get('content', 'N/A')[:20] for b in col[:3]]
        print(f"Column {i+1} (X≈{avg_x:.1f}): {len(col)} boxes - {', '.join(contents)}...")
    
    # Detect containers
    print(f"\n[DETECT] Detecting container relationships...")
    containers = detect_containers(boxes, proximity_threshold=50)
    print(f"[OK] Found {len(containers)} potential containers")
    
    # Full analysis
    analysis = analyze_layout(boxes)
    
    return {
        'rows': rows,
        'columns': columns,
        'containers': containers,
        'full_analysis': analysis,
        'thresholds': {'y': y_threshold, 'x': x_threshold}
    }


def run_vlm_analysis(boxes: List[Dict], dimensions: Dict, image_path: str, 
                     provider: str = 'openrouter', model: str = 'qwen/qwen-2-vl-7b-instruct',
                     max_boxes: int = 100) -> Optional[Dict]:
    """
    Run VLM-based analysis using OpenRouter API.
    
    Args:
        boxes: List of bounding boxes
        dimensions: Image dimensions
        image_path: Path to image
        provider: 'openrouter' or 'openai'
        model: Model name (default: qwen/qwen-2-vl-7b-instruct for OpenRouter)
        max_boxes: Maximum number of boxes to send to VLM (default: 100)
    
    Returns:
        VLM analysis results or None if VLM not available
    """
    if not OPENROUTER_AVAILABLE and provider == 'openrouter':
        print(f"\n[WARN]  OpenRouter API not available, skipping VLM analysis")
        return None
    
    print(f"\n{'='*80}")
    print(f"VLM-BASED ANALYSIS (AI Understanding)")
    print(f"{'='*80}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Max boxes to analyze: {max_boxes}")
    
    # Import and run VLM test
    try:
        # Create a custom analysis with limited boxes
        from test_vlm_layout_understanding import (
            format_boxes_for_prompt,
            create_layout_analysis_prompt
        )
        
        print(f"[VLM] Sending data to VLM (analyzing {min(len(boxes), max_boxes)} of {len(boxes)} boxes)...")
        
        # Create custom prompt with limited boxes
        boxes_text = format_boxes_for_prompt(boxes, max_boxes=max_boxes)
        
        prompt = f"""You are a UI layout analysis expert. I will provide you with bounding box coordinates for detected UI elements.

**Image Dimensions**: {dimensions['width']}x{dimensions['height']}px

**Detected Bounding Boxes** (showing {min(len(boxes), max_boxes)} of {len(boxes)} total):
{boxes_text}

**Your Task**:
Analyze the layout and identify:

1. **Horizontal Rows**: Which boxes are horizontally aligned (similar Y-coordinates)?
2. **Vertical Columns**: Which boxes are vertically aligned (similar X-coordinates)?
3. **Specific Detection**: Find "finance", "news", "shopping", "In usd", "Stock forecast" - are they horizontally aligned?

Provide concise analysis focusing on spatial relationships."""
        
        # Call OpenRouter API
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("[X] Error: OPENROUTER_API_KEY not found")
            print("Set it with: export OPENROUTER_API_KEY=your_openrouter_key")
            print("Get your key from: https://openrouter.ai/keys")
            return None
        
        API_URL = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization":f"Bearer {api_key}",
            "Content-Type":"application/json",
            "HTTP-Referer":"https://github.com/i-sababishraq/omniParser-v2",
            "X-Title":"OmniParser Layout Validation"
        }
        
        # Try to load and encode image
        image_base64 = None
        if image_path and Path(image_path).exists():
            try:
                with open(image_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
                print("[OK] Image encoded for vision input")
            except Exception as e:
                print(f"[WARN] Could not encode image: {e}")
        
        # Prepare messages with vision support
        if image_base64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 1.0
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"[X] API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        raw_response = result['choices'][0]['message']['content']
        
        result = {
            'model': model,
            'raw_response': raw_response,
            'boxes_analyzed': min(len(boxes), max_boxes)
        }
        
        print(f"[OK] VLM analysis complete")
        print(f"Response length: {len(raw_response)} characters")
        
        return result
            
    except Exception as e:
        print(f"[NO] Error running VLM analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_analyses(coord_results: Dict, vlm_results: Optional[Dict], boxes: List[Dict]) -> Dict:
    """
    Compare coordinate-based and VLM-based analyses.
    
    Returns:
        Comparison metrics and findings
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: Coordinate vs VLM Analysis")
    print(f"{'='*80}")
    
    comparison = {
        'coordinate_analysis': {
            'rows': len(coord_results['rows']),
            'columns': len(coord_results['columns']),
            'containers': len(coord_results['containers'])
        },
        'vlm_analysis': {},
        'agreement': {}
    }
    
    # Coordinate results
    print(f"\n[DATA] Coordinate-Based Detection:")
    print(f"- Horizontal rows: {len(coord_results['rows'])}")
    print(f"- Vertical columns: {len(coord_results['columns'])}")
    print(f"- Containers: {len(coord_results['containers'])}")
    
    # VLM results (if available)
    if vlm_results:
        print(f"\n[VLM] VLM-Based Detection:")
        
        # Try to parse VLM response for structured data
        raw = vlm_results.get('raw_response', '')
        
        # Count mentions of horizontal/vertical in VLM response
        horizontal_mentions = raw.lower().count('horizontal')
        vertical_mentions = raw.lower().count('vertical')
        row_mentions = raw.count('Row ')
        column_mentions = raw.count('Column ')
        
        print(f"- Mentions of 'horizontal': {horizontal_mentions}")
        print(f"- Mentions of 'vertical': {vertical_mentions}")
        print(f"- Row identifications: {row_mentions}")
        print(f"- Column identifications: {column_mentions}")
        
        comparison['vlm_analysis'] = {
            'horizontal_mentions': horizontal_mentions,
            'vertical_mentions': vertical_mentions,
            'row_mentions': row_mentions,
            'column_mentions': column_mentions
        }
        
        # Check for specific example validation
        print(f"\n[TARGET] Specific Example Validation:")
        print(f"Looking for: 'finance', 'news', 'shopping', 'In usd', 'Stock forecast'")
        
        # Find these boxes in coordinate analysis
        # Expanded keyword list with synonyms and variations for better matching
        target_keywords = [
            'finance', 'financial', 'finances',
            'news', 'article', 'articles',
            'shopping', 'shop', 'shops', 'store',
            'in usd', 'usd', 'us dollar', 'dollar',
            'stock forecast', 'stock forecasts', 'forecast', 'forecasts', 
            'prediction', 'predictions', 'stock prediction'
        ]
        target_boxes = []
        
        for box in boxes:
            content = box.get('content', '').lower()
            if any(keyword in content for keyword in target_keywords):
                target_boxes.append(box)
                print(f"Found: '{box.get('content', 'N/A')}' at Y={box['coordinates']['center']['y']:.1f}")
        
        if target_boxes:
            # Check if they're in the same row (coordinate-based)
            avg_y = sum(b['coordinates']['center']['y'] for b in target_boxes) / len(target_boxes)
            y_variance = max(abs(b['coordinates']['center']['y'] - avg_y) for b in target_boxes)
            
            print(f"\n  Coordinate Analysis:")
            print(f"Average Y: {avg_y:.2f}")
            print(f"Max Y variance: {y_variance:.2f}px")
            print(f"Horizontally aligned: {'[OK] YES' if y_variance < 15 else '[NO] NO'}")
            
            # Check VLM detection (use expanded keywords for matching)
            vlm_detected = any(keyword in raw.lower() for keyword in target_keywords)
            print(f"\n  VLM Detection:")
            print(f"Mentioned these elements: {'[OK] YES' if vlm_detected else '[NO] NO'}")
            
            if 'horizontally aligned' in raw.lower() or 'navigation row' in raw.lower():
                print(f"Identified as horizontal row: [OK] YES")
            else:
                print(f"Identified as horizontal row: ? UNCLEAR")

            comparison['specific_example'] = {
                'found_boxes': len(target_boxes),
                'avg_y': avg_y,
                'y_variance': y_variance,
                'coord_aligned': y_variance < 15,
                'vlm_detected': vlm_detected
            }
    else:
        print(f"\n[WARN]  VLM analysis not available for comparison")
    
    return comparison


def visualize_detections(
    boxes: List[Dict],
    coord_results: Dict,
    image_path: str,
    output_dir: str,
    dimensions: Dict
):
    """
    Generate visual validation with annotated images.
    """
    if not PIL_AVAILABLE:
        print(f"\n[WARN]  PIL not available, skipping visualization")
        return
    
    print(f"\n{'='*80}")
    print(f"VISUAL VALIDATION")
    print(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Create separate visualizations for rows and columns
        
        # 1. Horizontal rows visualization
        print(f"\n Generating horizontal rows visualization...")
        img_rows = img.copy()
        draw_rows = ImageDraw.Draw(img_rows)
        
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FF8000', '#8000FF', '#00FF80', '#FF0080'
        ]
        
        for i, row in enumerate(coord_results['rows'][:10]):  # Top 10 rows
            color = colors[i % len(colors)]
            for box in row:
                pixels = box['coordinates']['pixels']
                draw_rows.rectangle(
                    [(pixels['x1'], pixels['y1']), (pixels['x2'], pixels['y2'])],
                    outline=color,
                    width=3
                )
        
        rows_output = os.path.join(output_dir, 'validation_horizontal_rows.jpg')
        # Convert RGBA to RGB if needed for JPEG
        if img_rows.mode == 'RGBA':
            img_rows = img_rows.convert('RGB')
        img_rows.save(rows_output)
        print(f"  [OK] Saved to: {rows_output}")
        
        # 2. Vertical columns visualization
        print(f"\n Generating vertical columns visualization...")
        img_cols = img.copy()
        draw_cols = ImageDraw.Draw(img_cols)
        
        for i, col in enumerate(coord_results['columns'][:10]):  # Top 10 columns
            color = colors[i % len(colors)]
            for box in col:
                pixels = box['coordinates']['pixels']
                draw_cols.rectangle(
                    [(pixels['x1'], pixels['y1']), (pixels['x2'], pixels['y2'])],
                    outline=color,
                    width=3
                )
        
        cols_output = os.path.join(output_dir, 'validation_vertical_columns.jpg')
        # Convert RGBA to RGB if needed for JPEG
        if img_cols.mode == 'RGBA':
            img_cols = img_cols.convert('RGB')
        img_cols.save(cols_output)
        print(f"  [OK] Saved to: {cols_output}")
        
        # 3. Combined visualization with row/column grid
        print(f"\n Generating combined grid visualization...")
        img_grid = img.copy()
        draw_grid = ImageDraw.Draw(img_grid)
        
        # Draw horizontal row lines
        for i, row in enumerate(coord_results['rows'][:20]):
            avg_y = sum(b['coordinates']['center']['y'] for b in row) / len(row)
            draw_grid.line([(0, avg_y), (dimensions['width'], avg_y)], fill='#FF0000', width=2)
        
        # Draw vertical column lines
        for i, col in enumerate(coord_results['columns'][:20]):
            avg_x = sum(b['coordinates']['center']['x'] for b in col) / len(col)
            draw_grid.line([(avg_x, 0), (avg_x, dimensions['height'])], fill='#0000FF', width=2)
        
        grid_output = os.path.join(output_dir, 'validation_grid_overlay.jpg')
        # Convert RGBA to RGB if needed for JPEG
        if img_grid.mode == 'RGBA':
            img_grid = img_grid.convert('RGB')
        img_grid.save(grid_output)
        print(f"[OK] Saved to: {grid_output}")
        
        print(f"\n[OK] All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"[NO] Error generating visualizations: {e}")


def save_validation_report(
    coord_results: Dict,
    vlm_results: Optional[Dict],
    comparison: Dict,
    output_path: str
):
    """Save comprehensive validation report as JSON."""
    report = {
        'coordinate_analysis': {
            'num_rows': len(coord_results['rows']),
            'num_columns': len(coord_results['columns']),
            'num_containers': len(coord_results['containers']),
            'thresholds': coord_results['thresholds'],
            'rows_summary': [
                {
                    'row_id': i,
                    'num_boxes': len(row),
                    'avg_y': sum(b['coordinates']['center']['y'] for b in row) / len(row),
                    'sample_content': [b.get('content', 'N/A')[:30] for b in row[:3]]
                }
                for i, row in enumerate(coord_results['rows'][:20])
            ],
            'columns_summary': [
                {
                    'column_id': i,
                    'num_boxes': len(col),
                    'avg_x': sum(b['coordinates']['center']['x'] for b in col) / len(col),
                    'sample_content': [b.get('content', 'N/A')[:30] for b in col[:3]]
                }
                for i, col in enumerate(coord_results['columns'][:20])
            ]
        },
        'vlm_analysis': vlm_results if vlm_results else None,
        'comparison': comparison
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Validation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate layout detection (both horizontal and vertical)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation with VLM comparison (uses Qwen 2.5 VL from Hugging Face)
  python validate_layout_detection.py \\
      --input results_samples/single/demo_image_enhanced.json \\
      --image imgs/demo_image.jpg \\
      --output-dir results_samples/validation \\
      --use-vlm

  # Coordinate-only validation (no VLM)
  python validate_layout_detection.py \\
      --input results_samples/single/demo_image_enhanced.json \\
      --image imgs/demo_image.jpg \\
      --output-dir results_samples/validation

  # Custom thresholds
  python validate_layout_detection.py \\
      --input results_samples/single/demo_image_enhanced.json \\
      --image imgs/demo_image.jpg \\
      --y-threshold 15 \\
      --x-threshold 20
        """
    )
    
    parser.add_argument('--input', required=True, help='Path to enhanced JSON file')
    parser.add_argument('--image', required=True, help='Path to original image')
    parser.add_argument('--output-dir', default='results_samples/validation',
                        help='Output directory for validation results')
    parser.add_argument('--y-threshold', type=int, default=10,
                        help='Y-axis threshold for horizontal alignment (pixels)')
    parser.add_argument('--x-threshold', type=int, default=10,
                        help='X-axis threshold for vertical alignment (pixels)')
    parser.add_argument('--use-vlm', action='store_true',
                        help='Run VLM analysis for comparison')
    parser.add_argument('--vlm-provider', default='openrouter', choices=['openrouter', 'openai'],
                        help='VLM provider to use (default: openrouter)')
    parser.add_argument('--vlm-model', default='qwen/qwen-2-vl-7b-instruct',
                        help='VLM model to use (default: qwen/qwen-2-vl-7b-instruct for OpenRouter)')
    parser.add_argument('--vlm-max-boxes', type=int, default=100,
                        help='Maximum number of boxes to send to VLM (default: 100, helps avoid token limits)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    boxes, dimensions, image_path = load_data(args.input)
    
    # Use provided image path if different
    if args.image:
        image_path = args.image
    
    # Run coordinate-based analysis
    coord_results = run_coordinate_analysis(
        boxes,
        y_threshold=args.y_threshold,
        x_threshold=args.x_threshold
    )
    
    # Run VLM analysis if requested
    vlm_results = None
    if args.use_vlm:
        vlm_results = run_vlm_analysis(
            boxes, dimensions, image_path, 
            provider=args.vlm_provider,
            model=args.vlm_model,
            max_boxes=args.vlm_max_boxes
        )
    
    # Compare analyses
    comparison = compare_analyses(coord_results, vlm_results, boxes)
    
    # Generate visualizations
    visualize_detections(boxes, coord_results, image_path, args.output_dir, dimensions)
    
    # Save validation report
    report_path = os.path.join(args.output_dir, 'validation_report.json')
    save_validation_report(coord_results, vlm_results, comparison, report_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n[DATA] Summary:")
    print(f"- Coordinate-based rows: {len(coord_results['rows'])}")
    print(f"- Coordinate-based columns: {len(coord_results['columns'])}")
    print(f"- Validation visualizations: {args.output_dir}")
    print(f"- Full report: {report_path}")

    if vlm_results:
        print(f"- VLM analysis: [OK] Completed")
    else:
        print(f"- VLM analysis: ⊘ Skipped")
    print(f"\n[OK] All validation results saved successfully!")


if __name__ == '__main__':
    main()
