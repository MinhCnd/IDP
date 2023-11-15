import json
from idp.annotations.annotation_utils import ls_annotations_to_layoutlmv3


def test_ls_annotations_to_layoutlmv3():
    LOCAL_IMAGE_PATH = "tests\\"
    f = open("tests/testData.json", "r")
    data = json.load(f)
    result = ls_annotations_to_layoutlmv3(data[0], LOCAL_IMAGE_PATH)
    image = result.pop("image")

    assert result == {
        "tokens": ["0.00", "36.5", "49.83", "23.15", "109.48", "109.48"],
        "bboxes": [
            [861, 395, 899, 408],
            [856, 496, 897, 510],
            [855, 509, 899, 522],
            [854, 521, 896, 532],
            [849, 533, 897, 544],
            [848, 618, 898, 635],
        ],
        "ner_tags": [
            "B-BALANCE_STILL_OWING",
            "B-WATER_CONSUMPTION",
            "B-WASTEWATER_CONSUMPTION",
            "B-WASTEWATER_FIXED",
            "B-BALANCE_CURRENT_CHARGES",
            "B-TOTAL_DUE",
        ],
    }

    assert image.width == 1000 and image.height == 750 and image.mode == "RGB"
