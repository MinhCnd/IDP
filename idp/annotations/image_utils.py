from PIL import Image
from math import floor


def normalize_image_for_layoutlmv3(
    image: Image.Image, normalizedLength: float
) -> Image.Image:
    if image.width >= image.height:
        return image.resize(
            (normalizedLength, floor((image.height / image.width) * normalizedLength))
        )
    else:
        return image.resize(
            (floor((image.width / image.height) * normalizedLength), normalizedLength)
        )
