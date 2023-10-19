from idp.model.BoundingBox import BoundingBox
from math import floor


def label_studio_bbx_to_lmv3(x, y, width, height):
    """
    Convert label studio bounding box to lmv3 format
    """
    left = int(floor(x * 10))
    top = int(floor(y * 10))
    right = int(floor(left + width * 10))
    bot = int(floor(top + height * 10))
    print(left, top, right, bot)
    return [left, top, right, bot]


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]
