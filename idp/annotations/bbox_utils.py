from math import floor
from typing import List


def label_studio_bbx_to_lmv3(
    x: float, y: float, width: float, height: float
) -> List[int]:
    """
    Convert bounding box data from Label Studio to LayoutLMv3 format

    Input:
    x - X-pos of top left corner of bounding box, max 100
    y - Y-pos of top left corner of bounding box, max 100
    width - width of bounding box, max 100
    height - height of bounding box, max 100
    """
    return [
        int(floor(x * 10)),
        int(floor(y * 10)),
        int(floor((x + width) * 10)),
        int(floor((y + height) * 10)),
    ]


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
        int(floor((bbox[3] / height) * 1000)),
    ]


def merge_box_extremes(box_list: List[List[int]]) -> List[int]:
    """Merge box extremes
    Input:
    box_list - list of boxes to be merged, format(left, top, right, bottom)

    Output:
    merged_box - box formed from the extremes of the input boxes
    """
    if not box_list:
        return []

    for box in box_list:
        if len(box) != 4:
            raise ValueError("Each box must have 4 values")

    left_coords = [box[0] for box in box_list]
    top_coords = [box[1] for box in box_list]
    right_coords = [box[2] for box in box_list]
    bottom_coords = [box[3] for box in box_list]
    return [min(left_coords), min(top_coords), max(right_coords), max(bottom_coords)]


def is_box_a_within_box_b(box_a: List[int], box_b: List[int]) -> bool:
    """
    Check if bounds of box_a is within _box_b

    Inputs:
    box_a - first box, format(left, top, right bottom)
    box_b - second box, format(left, top, right bottom)

    Output:
    True of box_a bounds within box_b, False otherwise
    """
    left_a, top_a, right_a, bottom_a = box_a
    left_b, top_b, right_b, bottom_b = box_b

    a_within_b_horizontal = left_b <= left_a and right_b >= right_a
    a_within_b_vertical = top_b <= top_a and bottom_b >= bottom_a

    return a_within_b_horizontal and a_within_b_vertical
