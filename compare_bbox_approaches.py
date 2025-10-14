#!/usr/bin/env python3
"""
Compare Layout Detection Approaches: Enhanced Layout Analysis vs Pure VLM

This script compares two approaches for UI layout detection:

1. Enhanced Layout Analysis (Multi-layer approach):
   - Computer Vision Layer: OmniParser (Florence + YOLO) for precise bounding boxes
   - Spatial Analysis Layer: layout_detector for algorithmic grouping (rows, columns)
   - Semantic Understanding Layer: VLM for hierarchy, relationships, and patterns

2. Pure VLM (Vision-Language Model only):
   - VLM directly analyzes screenshot and predicts all element positions
   - No pre-computed detection or spatial preprocessing

Usage:
    # Full comparison
    python compare_bbox_approaches.py --image imgs/demo_image.jpg --output-dir comparison_results
    
    # Enhanced Layout Analysis only (skip Pure VLM)
    python compare_bbox_approaches.py --image imgs/demo_image.jpg --skip-vlm
    
    # Without VLM semantic layer (CV + spatial only)
    python compare_bbox_approaches.py --image imgs/demo_image.jpg --skip-hybrid-vlm
"""

import json
import argparse
import os
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, visualizations will be skipped")

load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def format_boxes_for_prompt(boxes: List[Dict], max_boxes: int = 100) -> str:
    """Format boxes as text for VLM prompt."""
    limited_boxes = boxes[:max_boxes]
    lines = []
    for idx, box in enumerate(limited_boxes):
        coords = box.get('coordinates', {}).get('pixels', {})
        content = box.get('content', '').strip()
        box_type = box.get('type', 'unknown')
        
        if content:
            lines.append(f"Box {idx}: [{box_type}] '{content}' at ({coords.get('x1', 0)}, {coords.get('y1', 0)}) to ({coords.get('x2', 0)}, {coords.get('y2', 0)})")
        else:
            lines.append(f"Box {idx}: [{box_type}] at ({coords.get('x1', 0)}, {coords.get('y1', 0)}) to ({coords.get('x2', 0)}, {coords.get('y2', 0)})")
    
    return '\n'.join(lines)


def run_vlm_semantic_analysis(image_path: str, boxes: List[Dict], dimensions: Dict, 
                               model: str, max_boxes: int = 100) -> Optional[Dict]:
    """
    Use VLM to analyze semantic relationships in the UI.
    
    Args:
        image_path: Path to screenshot
        boxes: Pre-detected bounding boxes from OmniParser
        dimensions: Image dimensions
        model: VLM model to use
        max_boxes: Max boxes to send to VLM
    
    Returns:
        Dict with VLM semantic analysis results
    """
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("[X] Error: OPENROUTER_API_KEY not found in environment")
        return None
    
    # Encode image
    try:
        image_base64 = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"[X] Error encoding image: {e}")
        return None
    
    # Format boxes for prompt
    boxes_text = format_boxes_for_prompt(boxes, max_boxes)
    
    # Create semantic analysis prompt
    prompt = f"""You are a UI layout expert. I have detected {len(boxes)} UI elements using computer vision. Analyze the semantic relationships and layout structure.

**Image Dimensions**: {dimensions['width']}x{dimensions['height']}px

**Detected Elements** (showing first {min(len(boxes), max_boxes)}):
{boxes_text}

**Your Task**: Analyze the layout and provide structured insights:

1. **Semantic Groupings**: Identify logical UI sections (header, navigation, content, sidebar, footer, forms, cards, etc.)
   - What elements belong together semantically?
   - What are the visual/functional containers?

2. **Layout Hierarchy**: Describe the parent-child relationships
   - Which elements contain others?
   - What is the nesting structure?

3. **Navigation & Interaction**: Identify interactive elements
   - Navigation bars, menus, buttons
   - Links, inputs, controls
   - Call-to-action elements

4. **Content Sections**: Identify major content areas
   - News/articles, products, listings
   - Data visualizations, charts
   - User-generated content areas

5. **Layout Patterns**: What design patterns are used?
   - Flexbox row/column
   - Grid layout
   - Card-based design
   - Dashboard widgets

6. **Improvements**: Suggest how to group these elements better for code generation
   - What CSS classes would you assign?
   - How should components be structured?

Respond in valid JSON format:
{{
  "semantic_sections": [
    {{"name": "header", "boxes": [0, 1, 2], "description": "Top navigation bar"}}
  ],
  "hierarchy": {{"parent_child": [{{"parent": 5, "children": [6, 7, 8]}}]}},
  "layout_pattern": "flex-row with grid content",
  "suggestions": ["Group boxes 10-15 as navigation menu", "Boxes 20-25 form a product card"]
}}"""
    
    print(f"  Sending {min(len(boxes), max_boxes)} boxes to VLM for semantic analysis...")
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/i-sababishraq/omniParser-v2",
        "X-Title": "BBox Hybrid Approach"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 8192
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code != 200:
            print(f"  [X] API Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
        
        # Try to parse JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"  [X] Failed to parse API response: {e}")
            print(f"  Response (first 300 chars): {response.text[:300]}")
            return None
        
        result_text = result['choices'][0]['message']['content']
        print(f"  [OK] VLM response ({len(result_text)} characters)")
        
        # Try to extract JSON from response
        vlm_data = None
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
        elif "```" in result_text:
            json_start = result_text.find("```") + 3
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
        else:
            json_text = result_text.strip()
        
        try:
            vlm_data = json.loads(json_text)
            print(f"  [OK] Parsed semantic analysis")
        except json.JSONDecodeError as e:
            print(f"  [WARN] Could not parse VLM JSON, saving raw text")
            vlm_data = {"raw_analysis": result_text}
        
        return vlm_data
        
    except Exception as e:
        print(f"  [X] Error calling VLM: {e}")
        return None


def run_coordinate_approach(image_path: str, use_vlm: bool = True, model: str = "qwen/qwen-2-vl-7b-instruct", max_boxes: int = 100) -> Dict:
    """
    Run Enhanced Layout Analysis: Multi-layer approach combining CV, spatial, and semantic analysis.
    
    This approach uses three complementary layers:
    1. Computer Vision Layer: Precise bounding boxes from OmniParser (Florence + YOLO)
    2. Spatial Analysis Layer: Algorithmic grouping via layout_detector (rows, columns, containers)
    3. Semantic Understanding Layer: VLM analysis for hierarchy, relationships, and patterns
    
    Returns:
        Dict with detected boxes, spatial analysis, and semantic insights
    """
    print(f"\n{'='*80}")
    print("APPROACH 1: ENHANCED LAYOUT ANALYSIS (OmniParser + Spatial + Semantic)")
    print(f"{'='*80}")
    
    # Check if we have pre-computed results
    image_stem = Path(image_path).stem
    json_path = f"results_samples/single/{image_stem}_enhanced.json"
    
    if not Path(json_path).exists():
        print(f"[INFO] Enhanced JSON not found at {json_path}")
        print(f"[INFO] You need to run OmniParser first:")
        print(f"       ./run_inference.sh {image_path}")
        return None
    
    # Load pre-computed boxes
    with open(json_path) as f:
        data = json.load(f)
    
    boxes = data.get('parsed_content_list', [])
    dimensions = data.get('image_dimensions', {})
    
    print(f"[OK] Loaded {len(boxes)} boxes from OmniParser (precise detection)")
    print(f"[OK] Image dimensions: {dimensions['width']}x{dimensions['height']}px")
    
    # Analyze layout with layout_detector
    from util.layout_detector import detect_horizontal_rows, detect_vertical_columns
    
    rows = detect_horizontal_rows(boxes, y_threshold=10)
    columns = detect_vertical_columns(boxes, x_threshold=10)
    
    print(f"[OK] Detected {len(rows)} horizontal rows (spatial grouping)")
    print(f"[OK] Detected {len(columns)} vertical columns (spatial grouping)")
    
    # Add VLM semantic analysis
    vlm_analysis = None
    if use_vlm:
        print(f"\n[VLM] Adding semantic analysis with {model}...")
        vlm_analysis = run_vlm_semantic_analysis(image_path, boxes, dimensions, model, max_boxes)
    
    return {
        'approach': 'enhanced-layout-analysis',
        'method': 'Enhanced Layout Analysis (CV + Spatial + Semantic)',
        'num_boxes': len(boxes),
        'boxes': boxes,
        'rows': len(rows),
        'columns': len(columns),
        'dimensions': dimensions,
        'vlm_analysis': vlm_analysis,
        'row_details': rows,
        'column_details': columns
    }


def run_bare_vlm_approach(image_path: str, model: str = "qwen/qwen-2-vl-7b-instruct") -> Optional[Dict]:
    """
    Run bare VLM approach - ask VLM to identify elements and their positions directly.
    
    This approach relies entirely on the VLM to:
    1. Identify UI elements
    2. Estimate bounding box positions
    3. Understand layout structure
    
    No pre-computed detection or spatial analysis.
    
    Args:
        image_path: Path to screenshot
        model: VLM model to use
    
    Returns:
        Dict with VLM-predicted boxes and positions
    """
    print(f"\n{'='*80}")
    print("APPROACH 2: PURE VLM (Vision-Language Model Only)")
    print(f"{'='*80}")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("[X] Error: OPENROUTER_API_KEY not found in environment")
        return None
    
    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size
    print(f"[OK] Image dimensions: {width}x{height}px")
    
    # Encode image
    try:
        image_base64 = encode_image_to_base64(image_path)
        print("[OK] Image encoded for vision input")
    except Exception as e:
        print(f"[X] Error encoding image: {e}")
        return None
    
    # Create prompt for VLM to detect UI elements
    prompt = f"""Analyze this UI screenshot (size: {width}x{height} pixels) and identify all visible elements.

For each element, provide:
- Type (button/text/icon/image/input)
- Text content
- Bounding box: x1, y1 (top-left), x2, y2 (bottom-right) in pixels

Return ONLY valid JSON in this format:
{{
  "elements": [
    {{"id": 1, "type": "button", "content": "Click", "bbox": {{"x1": 10, "y1": 20, "x2": 100, "y2": 50}}}},
    {{"id": 2, "type": "text", "content": "Hello", "bbox": {{"x1": 10, "y1": 60, "x2": 80, "y2": 90}}}}
  ]
}}

Provide coordinates for as many UI elements as you can identify:"""
    
    print(f"[VLM] Sending request to {model}...")
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/i-sababishraq/omniParser-v2",
        "X-Title": "BBox Approach Comparison"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 16000,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
        
        if response.status_code != 200:
            print(f"[X] API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        # Try to parse JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"[X] Failed to parse API response as JSON: {e}")
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response text (first 500 chars): {response.text[:500]}")
            return None
        
        result_text = result['choices'][0]['message']['content']
        
        print(f"[OK] Received response ({len(result_text)} characters)")
        
        # Try to extract JSON - handle multiple formats
        vlm_data = None
        
        # Try markdown code block format
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
        elif "```" in result_text:
            # Try generic code block
            json_start = result_text.find("```") + 3
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
        else:
            # Try to find JSON directly
            json_text = result_text.strip()
        
        # Parse JSON
        try:
            vlm_data = json.loads(json_text)
            print(f"[OK] Parsed JSON: {len(vlm_data.get('elements', []))} elements detected")
        except json.JSONDecodeError as e:
            print(f"[WARN] Could not parse JSON: {e}")
            print(f"[DEBUG] Attempting to fix common JSON issues...")
            
            # Try to fix common issues
            json_text_fixed = json_text.replace("'", '"')  # Single to double quotes
            json_text_fixed = json_text_fixed.replace('\n', ' ')  # Remove newlines
            
            try:
                vlm_data = json.loads(json_text_fixed)
                print(f"[OK] Fixed and parsed JSON: {len(vlm_data.get('elements', []))} elements")
            except:
                print(f"[DEBUG] Could not fix JSON. Saving raw text.")
                print(f"[DEBUG] First 500 chars: {result_text[:500]}")
                vlm_data = {"raw_text": result_text, "elements": [], "parse_error": str(e)}
        
        return {
            'approach': 'pure-vlm',
            'method': f'Pure VLM ({model})',
            'num_boxes': len(vlm_data.get('elements', [])),
            'boxes': vlm_data.get('elements', []),
            'raw_response': result_text,
            'dimensions': {'width': width, 'height': height}
        }
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Dict with 'coordinates.pixels' or 'bbox' keys
        box2: Dict with similar structure
    
    Returns:
        IoU score (0-1)
    """
    # Extract coordinates from different formats
    if 'coordinates' in box1:
        x1_1 = box1['coordinates']['pixels']['x1']
        y1_1 = box1['coordinates']['pixels']['y1']
        x2_1 = box1['coordinates']['pixels']['x2']
        y2_1 = box1['coordinates']['pixels']['y2']
    elif 'bbox' in box1:
        x1_1 = box1['bbox']['x1']
        y1_1 = box1['bbox']['y1']
        x2_1 = box1['bbox']['x2']
        y2_1 = box1['bbox']['y2']
    else:
        return 0.0
    
    if 'coordinates' in box2:
        x1_2 = box2['coordinates']['pixels']['x1']
        y1_2 = box2['coordinates']['pixels']['y1']
        x2_2 = box2['coordinates']['pixels']['x2']
        y2_2 = box2['coordinates']['pixels']['y2']
    elif 'bbox' in box2:
        x1_2 = box2['bbox']['x1']
        y1_2 = box2['bbox']['y1']
        x2_2 = box2['bbox']['x2']
        y2_2 = box2['bbox']['y2']
    else:
        return 0.0
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def compare_approaches(coord_result: Dict, vlm_result: Dict) -> Dict:
    """
    Compare the two approaches and generate metrics.
    
    Returns:
        Comparison metrics dictionary
    """
    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    if not coord_result or not vlm_result:
        print("[X] Cannot compare - one or both approaches failed")
        return {}
    
    coord_boxes = coord_result['boxes']
    vlm_boxes = vlm_result['boxes']
    
    print(f"\n[COUNTS]")
    print(f"Coordinate approach detected: {len(coord_boxes)} elements")
    print(f"Bare VLM approach detected: {len(vlm_boxes)} elements")
    print(f"Difference: {abs(len(coord_boxes) - len(vlm_boxes))} elements")
    
    # Try to match boxes by content similarity
    matches = []
    for coord_box in coord_boxes:
        coord_content = coord_box.get('content', '').lower().strip()
        if not coord_content:
            continue
        
        for vlm_box in vlm_boxes:
            vlm_content = vlm_box.get('content', '').lower().strip()
            if not vlm_content:
                continue
            
            # Simple content matching
            if coord_content in vlm_content or vlm_content in coord_content:
                iou = calculate_iou(coord_box, vlm_box)
                matches.append({
                    'coord_box': coord_box,
                    'vlm_box': vlm_box,
                    'iou': iou,
                    'content': coord_content
                })
    
    print(f"\n[MATCHES]")
    print(f"Found {len(matches)} content-based matches")
    
    if matches:
        avg_iou = sum(m['iou'] for m in matches) / len(matches)
        print(f"Average IoU for matched boxes: {avg_iou:.3f}")
        
        high_iou = [m for m in matches if m['iou'] > 0.5]
        print(f"High-quality matches (IoU > 0.5): {len(high_iou)}")
    
    return {
        'coord_count': len(coord_boxes),
        'vlm_count': len(vlm_boxes),
        'matches': len(matches),
        'avg_iou': sum(m['iou'] for m in matches) / len(matches) if matches else 0.0,
        'high_quality_matches': len([m for m in matches if m['iou'] > 0.5]),
        'matched_elements': matches[:10]  # Save first 10 for inspection
    }


def visualize_comparison(image_path: str, coord_result: Dict, vlm_result: Dict, output_dir: str):
    """Generate side-by-side visualization of both approaches."""
    if not PIL_AVAILABLE:
        print("[WARN] PIL not available, skipping visualization")
        return
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        img = Image.open(image_path)
        
        # Coordinate approach visualization
        if coord_result and coord_result['boxes']:
            img_coord = img.copy()
            draw = ImageDraw.Draw(img_coord)
            
            for box in coord_result['boxes'][:50]:  # Limit to 50 for clarity
                pixels = box['coordinates']['pixels']
                draw.rectangle(
                    [(pixels['x1'], pixels['y1']), (pixels['x2'], pixels['y2'])],
                    outline='#00FF00',
                    width=2
                )
            
            output_path = os.path.join(output_dir, 'coordinate_approach.jpg')
            if img_coord.mode == 'RGBA':
                img_coord = img_coord.convert('RGB')
            img_coord.save(output_path)
            print(f"[OK] Saved: {output_path}")
        
        # VLM approach visualization
        if vlm_result and vlm_result['boxes']:
            img_vlm = img.copy()
            draw = ImageDraw.Draw(img_vlm)
            
            for box in vlm_result['boxes']:
                if 'bbox' in box:
                    bbox = box['bbox']
                    draw.rectangle(
                        [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                        outline='#FF0000',
                        width=2
                    )
            
            output_path = os.path.join(output_dir, 'vlm_approach.jpg')
            if img_vlm.mode == 'RGBA':
                img_vlm = img_vlm.convert('RGB')
            img_vlm.save(output_path)
            print(f"[OK] Saved: {output_path}")
        
        print(f"\n[OK] Visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"[X] Error generating visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Enhanced Layout Analysis vs Pure VLM approaches for layout detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--image', required=True,
                        help='Path to screenshot image')
    parser.add_argument('--output-dir', default='comparison_results',
                        help='Output directory for comparison results')
    parser.add_argument('--vlm-model', default='qwen/qwen-2-vl-7b-instruct',
                        help='VLM model to use')
    parser.add_argument('--vlm-max-boxes', type=int, default=100,
                        help='Max boxes to send to VLM for semantic analysis')
    parser.add_argument('--skip-vlm', action='store_true',
                        help='Skip VLM approaches (spatial analysis only)')
    parser.add_argument('--skip-hybrid-vlm', action='store_true',
                        help='Skip VLM in Enhanced Layout Analysis (CV + spatial only)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LAYOUT DETECTION APPROACH COMPARISON")
    print("="*80)
    print(f"Approach 1: Enhanced Layout Analysis (CV + Spatial + Semantic)")
    print(f"Approach 2: Pure VLM (Vision-Language Model Only)")
    print(f"Image: {args.image}")
    print(f"Output: {args.output_dir}")
    print(f"VLM Model: {args.vlm_model}")
    
    # Run Enhanced Layout Analysis (OmniParser + layout_detector + VLM)
    use_vlm_in_hybrid = not args.skip_vlm and not args.skip_hybrid_vlm
    coord_result = run_coordinate_approach(
        args.image, 
        use_vlm=use_vlm_in_hybrid, 
        model=args.vlm_model,
        max_boxes=args.vlm_max_boxes
    )
    
    # Run bare VLM approach
    vlm_result = None
    if not args.skip_vlm:
        vlm_result = run_bare_vlm_approach(args.image, model=args.vlm_model)
    
    # Compare approaches
    if coord_result and vlm_result:
        comparison = compare_approaches(coord_result, vlm_result)
        
        # Save comparison report
        report = {
            'image': args.image,
            'enhanced_layout_analysis': {
                'method': coord_result['method'],
                'num_elements': coord_result['num_boxes'],
                'rows': coord_result.get('rows', 0),
                'columns': coord_result.get('columns', 0),
                'vlm_analysis': coord_result.get('vlm_analysis') is not None,
                'vlm_semantic_insights': coord_result.get('vlm_analysis', {})
            },
            'pure_vlm_approach': {
                'method': vlm_result['method'],
                'num_elements': vlm_result['num_boxes'],
                'raw_response_length': len(vlm_result.get('raw_response', ''))
            },
            'comparison': comparison
        }
        
        report_path = os.path.join(args.output_dir, 'comparison_report.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Comparison report saved: {report_path}")
        
        # Generate visualizations
        visualize_comparison(args.image, coord_result, vlm_result, args.output_dir)
    elif coord_result:
        # Only Enhanced Layout Analysis ran
        print("\n[INFO] Only Enhanced Layout Analysis completed (Pure VLM skipped)")
        report = {
            'image': args.image,
            'enhanced_layout_analysis': {
                'method': coord_result['method'],
                'num_elements': coord_result['num_boxes'],
                'rows': coord_result.get('rows', 0),
                'columns': coord_result.get('columns', 0),
                'vlm_analysis': coord_result.get('vlm_analysis') is not None
            }
        }
        report_path = os.path.join(args.output_dir, 'enhanced_layout_analysis.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[OK] Enhanced Layout Analysis saved: {report_path}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
