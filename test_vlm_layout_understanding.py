#!/usr/bin/env python3
"""
VLM Layout Understanding Test

Tests if VLMs can understand UI layout hierarchy from screenshots with bounding boxes.

Uses OpenRouter API with Qwen 2 VL to analyze:
- Horizontal/vertical alignment
- Parent-child relationships
- Semantic sections (header, nav, content, footer)
- Layout patterns (grid, flex, absolute)
"""

import json
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

OPENROUTER_AVAILABLE = True


def load_enhanced_json(json_path: str) -> Dict:
    """Load enhanced JSON file with bounding boxes."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"[OK] Loaded {data.get('num_boxes', 0)} boxes from {json_path}")
    return data


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def format_boxes_for_prompt(boxes: List[Dict], max_boxes: int = 50) -> str:
    """
    Format bounding boxes into a readable text format for VLM.
    
    Args:
        boxes: List of box dictionaries
        max_boxes: Maximum boxes to include (to avoid token limits)
    
    Returns:
        Formatted string with box information
    """
    lines = []
    
    for i, box in enumerate(boxes[:max_boxes]):
        box_id = box.get('index_number', i)
        content = box.get('content', 'N/A')[:50]
        box_type = box.get('type', 'unknown')
        interactive = box.get('interactivity', False)
        
        center = box['coordinates']['center']
        pixels = box['coordinates']['pixels']
        size = box['coordinates']['size']
        
        lines.append(
            f"Box {box_id}: '{content}' | "
            f"Type: {box_type} | "
            f"Interactive: {interactive} | "
            f"Center: ({center['x']}, {center['y']}) | "
            f"Position: ({pixels['x1']}, {pixels['y1']}) to ({pixels['x2']}, {pixels['y2']}) | "
            f"Size: {size['width']}x{size['height']}px"
        )
    
    if len(boxes) > max_boxes:
        lines.append(f"\n... and {len(boxes) - max_boxes} more boxes")
    
    return '\n'.join(lines)


def create_layout_analysis_prompt(boxes: List[Dict], image_dimensions: Dict) -> str:
    """
    Create prompt for VLM to analyze layout hierarchy.
    
    Args:
        boxes: List of bounding boxes
        image_dimensions: Image width and height dict
    
    Returns:
        Prompt string for VLM
    """
    boxes_text = format_boxes_for_prompt(boxes, max_boxes=100)
    
    prompt = f"""You are a UI layout analysis expert. I will provide you with a screenshot and bounding box coordinates for all detected UI elements.

**Image Dimensions**: {image_dimensions['width']}x{image_dimensions['height']}px

**Detected Bounding Boxes**:
{boxes_text}

**Your Task**:
Analyze the layout and provide a structured hierarchy understanding. Specifically:

1. **Horizontal Rows**: Which boxes are horizontally aligned (similar Y-coordinates)? Group them into rows.
   - Example: "Row 1 (Y~35): Box 0 'New Tab', Box 1 'Search text'"

2. **Vertical Columns**: Which boxes are vertically aligned (similar X-coordinates)? Group them into columns.
   - Example: "Column 1 (X~100): Box 5 'Label 1', Box 12 'Label 2'"

3. **Semantic Sections**: Identify major UI sections based on position and content:
   - Header (top section with navigation/logo)
   - Navigation bar (horizontal list of links)
   - Content area (main body)
   - Sidebar (left or right column)
   - Footer (bottom section)

4. **Layout Patterns**: What layout pattern is used?
   - Horizontal flex row
   - Vertical flex column
   - Grid (multiple rows with similar column counts)
   - Absolute positioning
   - Mixed/complex

5. **Parent-Child Relationships**: Are there container elements that contain other elements?
   - Example: "Box 10 contains Box 11, 12, 13 (navigation menu)"

6. **Specific Detection**: Based on the content, find elements like "finance", "news", "shopping", "In usd", "Stock forecast".
   - Are they horizontally aligned?
   - Do they form a navigation row?
   - What is their Y-coordinate similarity?

**Output Format**:
Provide your analysis in JSON format with horizontal_rows, vertical_columns, semantic_sections, layout_pattern, special_detections, and parent_child_relationships.

**Important**: 
- Use the actual box IDs and content from the provided data
- Calculate Y/X coordinate similarities to determine alignment
- Consider boxes aligned if their Y-coordinates differ by <=10px (or X for vertical)
- Provide reasoning for your groupings
"""
    
    return prompt


def analyze_with_openrouter(image_path: str, boxes: List[Dict], image_dimensions: Dict, 
                      model: str = "qwen/qwen-2-vl-7b-instruct") -> Optional[Dict]:
    """
    Use OpenRouter API to analyze layout hierarchy.
    
    Args:
        image_path: Path to screenshot image
        boxes: List of bounding boxes
        image_dimensions: Image width and height
        model: OpenRouter model to use
    
    Returns:
        VLM analysis result as dict
    """
    if not OPENROUTER_AVAILABLE:
        print("Error: requests package not installed")
        return None
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        return None
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/i-sababishraq/omniParser-v2",
        "X-Title": "VLM Layout Understanding Test"
    }
    
    prompt = create_layout_analysis_prompt(boxes, image_dimensions)
    
    print(f"\n=== Sending to OpenRouter ({model}) ===")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Analyzing {len(boxes)} boxes...")
    
    try:
        image_base64 = None
        if image_path and Path(image_path).exists():
            try:
                image_base64 = encode_image_to_base64(image_path)
                print("  [OK] Image encoded for vision input")
            except Exception as e:
                print(f"  [WARN] Could not encode image: {e}")
        
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
            "max_tokens": 4096,
            "top_p": 1.0
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"  [X] API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
        
        result = response.json()
        result_text = result['choices'][0]['message']['content']
        
        print(f"\n[OK] Received response ({len(result_text)} characters)")
        
        analysis = {}
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
            try:
                analysis = json.loads(json_text)
                print("[OK] Parsed JSON response")
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse JSON: {e}")
                analysis = {"raw_text": result_text}
        else:
            analysis = {"raw_text": result_text}
        
        return analysis
        
    except Exception as e:
        print(f"  [X] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_with_openai(image_path: str, boxes: List[Dict], image_dimensions: Dict,
                        model: str = "gpt-4o") -> Optional[Dict]:
    """
    Use OpenAI API to analyze layout hierarchy.
    
    Args:
        image_path: Path to screenshot image
        boxes: List of bounding boxes
        image_dimensions: Image width and height
        model: OpenAI model to use
    
    Returns:
        VLM analysis result as dict
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Install with: pip install openai")
        return None
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return None
    
    client = OpenAI(api_key=api_key)
    image_base64 = encode_image_to_base64(image_path)
    prompt = create_layout_analysis_prompt(boxes, image_dimensions)
    
    print(f"\n=== Sending to OpenAI ({model}) ===")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Image: {image_path}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
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
            ],
            max_tokens=4096,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        print(f"\n[OK] Received response ({len(result_text)} characters)")
        
        analysis = {}
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            json_text = result_text[json_start:json_end].strip()
            try:
                analysis = json.loads(json_text)
                print("[OK] Parsed JSON response")
            except json.JSONDecodeError as e:
                print(f"[WARN] Could not parse JSON: {e}")
                analysis = {"raw_text": result_text}
        else:
            analysis = {"raw_text": result_text}
        
        return analysis
        
    except Exception as e:
        print(f"  [X] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_analysis_results(analysis: Dict):
    """Print analysis results in a readable format."""
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    if "horizontal_rows" in analysis:
        print("\n--- Horizontal Rows ---")
        for row in analysis["horizontal_rows"]:
            print(f"Row {row.get('row_id', '?')} (Y~{row.get('avg_y', '?')}):")
            print(f"  Boxes: {row.get('boxes', [])}")
            print(f"  Description: {row.get('description', 'N/A')}")
    
    if "vertical_columns" in analysis:
        print("\n--- Vertical Columns ---")
        for col in analysis["vertical_columns"]:
            print(f"Column {col.get('column_id', '?')} (X~{col.get('avg_x', '?')}):")
            print(f"  Boxes: {col.get('boxes', [])}")
            print(f"  Description: {col.get('description', 'N/A')}")
    
    if "semantic_sections" in analysis:
        print("\n--- Semantic Sections ---")
        for section_name, section_data in analysis["semantic_sections"].items():
            print(f"{section_name.capitalize()}:")
            print(f"  Boxes: {section_data.get('boxes', [])}")
            print(f"  Y-range: {section_data.get('y_range', [])}")
            print(f"  Description: {section_data.get('description', 'N/A')}")
    
    if "special_detections" in analysis:
        print("\n--- Special Detections ---")
        for detection_name, detection_data in analysis["special_detections"].items():
            print(f"{detection_name}:")
            for key, value in detection_data.items():
                print(f"  {key}: {value}")
    
    if "layout_pattern" in analysis:
        print(f"\n--- Overall Layout Pattern ---")
        print(f"Pattern: {analysis['layout_pattern']}")


def main():
    parser = argparse.ArgumentParser(
        description='Test VLM understanding of UI layout hierarchy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Path to enhanced JSON file')
    parser.add_argument('--image', required=True,
                        help='Path to original screenshot image')
    parser.add_argument('--provider', default='openrouter',
                        choices=['openrouter', 'openai'],
                        help='VLM provider to use (default: openrouter)')
    parser.add_argument('--model', 
                        help='Model name (default: qwen/qwen-2-vl-7b-instruct for openrouter, gpt-4o for openai)')
    parser.add_argument('--output', '-o',
                        help='Save analysis results to JSON file')
    
    args = parser.parse_args()
    
    if not args.model:
        args.model = 'qwen/qwen-2-vl-7b-instruct' if args.provider == 'openrouter' else 'gpt-4o'
    
    print("=== VLM Layout Understanding Test ===\n")
    
    data = load_enhanced_json(args.input)
    boxes = data['parsed_content_list']
    image_dimensions = data['image_dimensions']
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"[OK] Image: {args.image}")
    print(f"[OK] Provider: {args.provider}")
    print(f"[OK] Model: {args.model}")
    
    if args.provider == 'openrouter':
        analysis = analyze_with_openrouter(args.image, boxes, image_dimensions, args.model)
    elif args.provider == 'openai':
        analysis = analyze_with_openai(args.image, boxes, image_dimensions, args.model)
    else:
        print(f"Error: Unsupported provider: {args.provider}")
        return
    
    if not analysis:
        print("Error: No analysis results received")
        return
    
    print_analysis_results(analysis)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n[OK] Saved analysis to {args.output}")
    
    print("\n[OK] Analysis complete!")


if __name__ == '__main__':
    main()
