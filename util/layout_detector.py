"""
Layout Detection Module

Detects spatial relationships from OmniParser bounding boxes:
- Horizontal rows (elements aligned by Y-coordinate)
- Vertical columns (elements aligned by X-coordinate)
- Containers and groups (proximity-based clustering)
"""

from typing import List, Dict, Tuple, Optional
import math


def detect_horizontal_rows(boxes: List[Dict], y_threshold: int = 10) -> List[List[Dict]]:
    """
    Group boxes into horizontal rows based on Y-coordinate similarity.
    
    Args:
        boxes: List of enhanced box dictionaries with 'coordinates' key
        y_threshold: Max pixel difference for same row (default: 10px)
    
    Returns:
        List of rows, each row is a list of boxes sorted by X-coordinate
    
    Example:
        >>> boxes = [
        ...     {'content': 'finance', 'coordinates': {'center': {'x': 1571, 'y': 432}}},
        ...     {'content': 'news', 'coordinates': {'center': {'x': 1798, 'y': 432}}},
        ...     {'content': 'shopping', 'coordinates': {'center': {'x': 2024, 'y': 435}}}
        ... ]
        >>> rows = detect_horizontal_rows(boxes, y_threshold=10)
        >>> len(rows[0])  # All three in same row
        3
    """
    if not boxes:
        return []
    
    # Sort boxes by center Y-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: b['coordinates']['center']['y'])
    
    rows = []
    current_row = []
    current_y = None
    
    for box in sorted_boxes:
        center_y = box['coordinates']['center']['y']
        
        if current_y is None:
            # First box - start new row
            current_y = center_y
            current_row = [box]
        elif abs(center_y - current_y) <= y_threshold:
            # Within threshold - add to current row
            current_row.append(box)
            # Update average Y for better grouping
            current_y = sum(b['coordinates']['center']['y'] for b in current_row) / len(current_row)
        else:
            # Outside threshold - finalize current row, start new one
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b['coordinates']['center']['x']))
            current_y = center_y
            current_row = [box]
    
    # Add final row
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b['coordinates']['center']['x']))
    
    return rows


def detect_vertical_columns(boxes: List[Dict], x_threshold: int = 10) -> List[List[Dict]]:
    """
    Group boxes into vertical columns based on X-coordinate similarity.
    
    Args:
        boxes: List of enhanced box dictionaries with 'coordinates' key
        x_threshold: Max pixel difference for same column (default: 10px)
    
    Returns:
        List of columns, each column is a list of boxes sorted by Y-coordinate
    
    Example:
        >>> boxes = [
        ...     {'content': 'Label 1', 'coordinates': {'center': {'x': 100, 'y': 50}}},
        ...     {'content': 'Label 2', 'coordinates': {'center': {'x': 102, 'y': 100}}},
        ...     {'content': 'Label 3', 'coordinates': {'center': {'x': 98, 'y': 150}}}
        ... ]
        >>> cols = detect_vertical_columns(boxes, x_threshold=10)
        >>> len(cols[0])  # All three in same column
        3
    """
    if not boxes:
        return []
    
    # Sort boxes by center X-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: b['coordinates']['center']['x'])
    
    columns = []
    current_column = []
    current_x = None
    
    for box in sorted_boxes:
        center_x = box['coordinates']['center']['x']
        
        if current_x is None:
            # First box - start new column
            current_x = center_x
            current_column = [box]
        elif abs(center_x - current_x) <= x_threshold:
            # Within threshold - add to current column
            current_column.append(box)
            # Update average X for better grouping
            current_x = sum(b['coordinates']['center']['x'] for b in current_column) / len(current_column)
        else:
            # Outside threshold - finalize current column, start new one
            if current_column:
                columns.append(sorted(current_column, key=lambda b: b['coordinates']['center']['y']))
            current_x = center_x
            current_column = [box]
    
    # Add final column
    if current_column:
        columns.append(sorted(current_column, key=lambda b: b['coordinates']['center']['y']))
    
    return columns


def is_contained(inner: Dict, outer: Dict) -> bool:
    """
    Check if inner box is fully contained within outer box.
    
    Args:
        inner: Box dictionary with 'coordinates.pixels' key
        outer: Box dictionary with 'coordinates.pixels' key
    
    Returns:
        True if inner is fully contained within outer
    
    Example:
        >>> inner = {'coordinates': {'pixels': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}}}
        >>> outer = {'coordinates': {'pixels': {'x1': 50, 'y1': 50, 'x2': 250, 'y2': 250}}}
        >>> is_contained(inner, outer)
        True
    """
    inner_coords = inner['coordinates']['pixels']
    outer_coords = outer['coordinates']['pixels']
    
    return (outer_coords['x1'] <= inner_coords['x1'] and
            outer_coords['y1'] <= inner_coords['y1'] and
            outer_coords['x2'] >= inner_coords['x2'] and
            outer_coords['y2'] >= inner_coords['y2'])


def is_nearby(box1: Dict, box2: Dict, threshold: float) -> bool:
    """
    Check if two boxes are within proximity threshold.
    
    Args:
        box1: Box dictionary with 'coordinates.center' key
        box2: Box dictionary with 'coordinates.center' key
        threshold: Distance threshold in pixels
    
    Returns:
        True if boxes are within threshold distance
    
    Example:
        >>> box1 = {'coordinates': {'center': {'x': 100, 'y': 100}}}
        >>> box2 = {'coordinates': {'center': {'x': 110, 'y': 105}}}
        >>> is_nearby(box1, box2, threshold=20)
        True
    """
    center1 = box1['coordinates']['center']
    center2 = box2['coordinates']['center']
    
    distance = math.sqrt((center1['x'] - center2['x'])**2 + (center1['y'] - center2['y'])**2)
    return distance <= threshold


def detect_containers(boxes: List[Dict], proximity_threshold: float = 50) -> Tuple[List[Dict], List[List[Dict]]]:
    """
    Detect groups/containers based on spatial proximity and containment.
    
    Args:
        boxes: List of enhanced box dictionaries
        proximity_threshold: Distance threshold for grouping (default: 50px)
    
    Returns:
        Tuple of (containers, groups)
        - containers: List of dicts with 'parent' and 'children' keys
        - groups: List of box lists grouped by proximity
    
    Example:
        >>> boxes = [
        ...     {'index': 1, 'coordinates': {...}},
        ...     {'index': 2, 'coordinates': {...}},
        ...     {'index': 3, 'coordinates': {...}}
        ... ]
        >>> containers, groups = detect_containers(boxes)
    """
    # Check for containment relationships
    containers = []
    for i, box1 in enumerate(boxes):
        children = []
        for j, box2 in enumerate(boxes):
            if i != j and is_contained(box2, box1):
                # Avoid nested containment (only direct children)
                is_direct_child = True
                for k, box3 in enumerate(boxes):
                    if k != i and k != j and is_contained(box2, box3) and is_contained(box3, box1):
                        is_direct_child = False
                        break
                if is_direct_child:
                    children.append(box2)
        
        if children:
            containers.append({'parent': box1, 'children': children})
    
    # Check for proximity groups
    groups = []
    visited = set()
    
    for i, box in enumerate(boxes):
        if i in visited:
            continue
        
        group = [box]
        visited.add(i)
        
        # Find all boxes nearby this one
        for j, other in enumerate(boxes):
            if j in visited:
                continue
            if is_nearby(box, other, proximity_threshold):
                group.append(other)
                visited.add(j)
        
        # Only add as group if more than one element
        if len(group) > 1:
            groups.append(group)
    
    return containers, groups


def calculate_spacing(boxes: List[Dict]) -> Dict[str, float]:
    """
    Calculate spacing statistics for a list of boxes.
    
    Useful for determining gap sizes between elements in a row/column.
    
    Args:
        boxes: List of boxes (should be sorted by position)
    
    Returns:
        Dict with 'mean_gap', 'median_gap', 'min_gap', 'max_gap'
    
    Example:
        >>> boxes = [
        ...     {'coordinates': {'center': {'x': 100, 'y': 50}}},
        ...     {'coordinates': {'center': {'x': 150, 'y': 50}}},
        ...     {'coordinates': {'center': {'x': 200, 'y': 50}}}
        ... ]
        >>> spacing = calculate_spacing(boxes)
        >>> spacing['mean_gap']
        50.0
    """
    if len(boxes) < 2:
        return {'mean_gap': 0, 'median_gap': 0, 'min_gap': 0, 'max_gap': 0}
    
    # Calculate gaps between consecutive boxes
    gaps = []
    for i in range(len(boxes) - 1):
        center1 = boxes[i]['coordinates']['center']
        center2 = boxes[i + 1]['coordinates']['center']
        gap = math.sqrt((center2['x'] - center1['x'])**2 + (center2['y'] - center1['y'])**2)
        gaps.append(gap)
    
    gaps.sort()
    n = len(gaps)
    median = gaps[n // 2] if n % 2 == 1 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2
    
    return {
        'mean_gap': sum(gaps) / len(gaps),
        'median_gap': median,
        'min_gap': min(gaps),
        'max_gap': max(gaps)
    }


def detect_layout_type(boxes: List[Dict]) -> str:
    """
    Detect the layout type of a group of boxes.
    
    Args:
        boxes: List of box dictionaries
    
    Returns:
        One of: 'horizontal', 'vertical', 'grid', 'scattered'
    
    Example:
        >>> boxes = [
        ...     {'coordinates': {'center': {'x': 100, 'y': 50}}},
        ...     {'coordinates': {'center': {'x': 200, 'y': 52}}},
        ...     {'coordinates': {'center': {'x': 300, 'y': 48}}}
        ... ]
        >>> detect_layout_type(boxes)
        'horizontal'
    """
    if len(boxes) < 2:
        return 'scattered'
    
    # Calculate variance in X and Y coordinates
    xs = [b['coordinates']['center']['x'] for b in boxes]
    ys = [b['coordinates']['center']['y'] for b in boxes]
    
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    
    x_variance = sum((x - x_mean) ** 2 for x in xs) / len(xs)
    y_variance = sum((y - y_mean) ** 2 for y in ys) / len(ys)
    
    # Low Y variance = horizontal layout
    # Low X variance = vertical layout
    # Both high = grid or scattered
    
    if y_variance < 100 and x_variance > 1000:
        return 'horizontal'
    elif x_variance < 100 and y_variance > 1000:
        return 'vertical'
    elif x_variance > 1000 and y_variance > 1000:
        # Check if it forms a grid
        rows = detect_horizontal_rows(boxes, y_threshold=20)
        if len(rows) > 1:
            # Check if rows have similar number of elements
            row_lengths = [len(row) for row in rows]
            if max(row_lengths) - min(row_lengths) <= 1:
                return 'grid'
        return 'scattered'
    else:
        return 'scattered'


def analyze_layout(boxes: List[Dict], y_threshold: int = 10, x_threshold: int = 10) -> Dict:
    """
    Comprehensive layout analysis of a list of boxes.
    
    Args:
        boxes: List of enhanced box dictionaries
        y_threshold: Threshold for horizontal alignment
        x_threshold: Threshold for vertical alignment
    
    Returns:
        Dict with complete layout analysis:
        {
            'rows': [...],
            'columns': [...],
            'containers': [...],
            'groups': [...],
            'layout_type': 'horizontal'|'vertical'|'grid'|'scattered',
            'num_boxes': int
        }
    
    Example:
        >>> boxes = load_boxes_from_json('demo_image_enhanced.json')
        >>> layout = analyze_layout(boxes)
        >>> print(f"Detected {len(layout['rows'])} rows")
    """
    rows = detect_horizontal_rows(boxes, y_threshold=y_threshold)
    columns = detect_vertical_columns(boxes, x_threshold=x_threshold)
    containers, groups = detect_containers(boxes)
    layout_type = detect_layout_type(boxes)
    
    return {
        'rows': rows,
        'columns': columns,
        'containers': containers,
        'groups': groups,
        'layout_type': layout_type,
        'num_boxes': len(boxes)
    }


# ============================================================================
# OPTIONAL TEST BLOCK - Can be safely removed (lines 382-411)
# This block only runs when executing: python util/layout_detector.py
# It does NOT affect imports or usage in other files
# ============================================================================
if __name__ == '__main__':
    # Simple test
    test_boxes = [
        {
            'content': 'finance',
            'coordinates': {
                'center': {'x': 1571, 'y': 432},
                'pixels': {'x1': 1458, 'y1': 389, 'x2': 1685, 'y2': 475}
            }
        },
        {
            'content': 'news',
            'coordinates': {
                'center': {'x': 1798, 'y': 432},
                'pixels': {'x1': 1717, 'y1': 389, 'x2': 1879, 'y2': 475}
            }
        },
        {
            'content': 'shopping',
            'coordinates': {
                'center': {'x': 2024, 'y': 435},
                'pixels': {'x1': 1911, 'y1': 389, 'x2': 2138, 'y2': 475}
            }
        }
    ]
    
    print("Testing layout detection...")
    rows = detect_horizontal_rows(test_boxes, y_threshold=10)
    print(f"✓ Detected {len(rows)} row(s)")
    print(f"✓ First row has {len(rows[0])} elements: {[b['content'] for b in rows[0]]}")
    
    layout = analyze_layout(test_boxes)
    print(f"✓ Layout type: {layout['layout_type']}")
    print("✓ Layout detector tests passed!")
# ============================================================================
# END OF OPTIONAL TEST BLOCK
# ============================================================================
