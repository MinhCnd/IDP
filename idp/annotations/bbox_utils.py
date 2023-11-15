from math import floor
from typing import List

def unnormalize_box(bbox: List[int], width: int, height: int) -> List[int]:
    return [
        int(floor(width * (bbox[0] / 1000))),
        int(floor(height * (bbox[1] / 1000))),
        int(floor(width * (bbox[2] / 1000))),
        int(floor(height * (bbox[3] / 1000))),
    ]

def normalize_box(bbox: List[int], width: int, height: int) -> List[int]:
    return [
        int(floor((bbox[0] / width) * 1000)),
        int(floor((bbox[1] / height) * 1000)),
        int(floor((bbox[2] / width) * 1000)),
        int(floor((bbox[3] / height) * 1000))
    ]

def merge_box_extremes(box_list: List[List[int]]) -> List[int]:
    """ Merge box extremes
    Input:
    box_list - list of boxes to be merged
    
    Output:
    merged_box - box formed from the extremes of the input boxes
    """
    if not box_list:
        return []
    left_coords = [box[0] for box in box_list]
    top_coords = [box[1] for box in box_list]
    right_coords = [box[2] for box in box_list]
    bottom_coords = [box[3] for box in box_list]
    return [min(left_coords),min(top_coords), max(right_coords), max(bottom_coords)]