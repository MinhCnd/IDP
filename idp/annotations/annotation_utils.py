from typing import Dict, List, Union
from enum import Enum
from PIL import Image
import re
from pathlib import Path

from idp.annotations.bbox_utils import label_studio_bbx_to_lmv3
from idp.annotations.image_utils import normalize_image_for_layoutlmv3


class Classes(Enum):
    BALANCE_STILL_OWING = 1
    WATER_CONSUMPTION = 2
    WASTEWATER_CONSUMPTION = 3
    WASTEWATER_FIXED = 4
    BALANCE_CURRENT_CHARGES = 5
    TOTAL_DUE = 6
    WATER_CONSUMPTION_DETAILS = 7  # IGNORE
    WASTEWATER_CONSUMPTION_DETAILS = 8  # IGNORE
    WASTEWATER_FIXED_DETAILS = 9  # IGNORE


LABEL_STR_TO_CLASS_MAP = {
    "balance_still_owing": Classes.BALANCE_STILL_OWING,
    "water_consumption": Classes.WATER_CONSUMPTION,
    "wastewater_consumption": Classes.WASTEWATER_CONSUMPTION,
    "wastewater_fixed": Classes.WASTEWATER_FIXED,
    "balance_current_charges": Classes.BALANCE_CURRENT_CHARGES,
    "total_due": Classes.TOTAL_DUE,
}

CLASS_TO_LABEL_MAP = {
    Classes.BALANCE_STILL_OWING: "B-BALANCE_STILL_OWING",
    Classes.WATER_CONSUMPTION: "B-WATER_CONSUMPTION",
    Classes.WASTEWATER_CONSUMPTION: "B-WASTEWATER_CONSUMPTION",
    Classes.WASTEWATER_FIXED: "B-WASTEWATER_FIXED",
    Classes.BALANCE_CURRENT_CHARGES: "B-BALANCE_CURRENT_CHARGES",
    Classes.TOTAL_DUE: "B-TOTAL_DUE",
}


def label_to_iob2(label: str) -> Union[str, None]:
    if label in list(LABEL_STR_TO_CLASS_MAP.keys()):
        return f"B-{LABEL_STR_TO_CLASS_MAP[label].name}"
    else:
        return None


def label_to_ner_tag(label: str) -> Union[str, None]:
    if label in list(LABEL_STR_TO_CLASS_MAP.keys()):
        cls = LABEL_STR_TO_CLASS_MAP[label]
        return CLASS_TO_LABEL_MAP[cls]
    else:
        return None


def get_img_src(label_studio_path: str, new_prefix: str) -> Path:
    fileName = re.findall(r"\d+-\d+.png", label_studio_path)[0]
    return Path(new_prefix, fileName)


def ls_annotations_to_layoutlmv3(
    annotations: List[Dict], local_image_path: str
) -> Dict:
    """
    Convert Label Studio annotations to LayoutLMv3 format

    Arguments:
    annotations -- list of Label Studio annotations
    local_image_path -- local path to image folder
    """
    tokens = []
    bboxes = []
    ner_tags = []

    transcriptions = {}

    for result in annotations["annotations"][0]["result"]:
        result_id = result["id"]

        if result_id not in transcriptions.keys():
            transcriptions[result_id] = {}

        if result["from_name"] == "transcription":
            transcriptions[result_id]["value"] = result["value"]
        elif result["from_name"] == "label":
            transcriptions[result_id]["label"] = result["value"]["labels"][0]

    # # TODO: REMOVE IGNORE ON SENTENCES
    def removeIgnoredLabels(k_v_pairs):
        _, value = k_v_pairs
        return value["label"] in list(LABEL_STR_TO_CLASS_MAP.keys())

    transcriptions = dict(filter(removeIgnoredLabels, transcriptions.items()))

    tokens = [
        transcription["value"]["text"][0] for transcription in transcriptions.values()
    ]
    bboxes = [
        label_studio_bbx_to_lmv3(
            transcription["value"]["x"],
            transcription["value"]["y"],
            transcription["value"]["width"],
            transcription["value"]["height"],
        )
        for transcription in transcriptions.values()
    ]
    ner_tags = [
        label_to_ner_tag(transcription["label"])
        for transcription in transcriptions.values()
    ]

    image_path = get_img_src(annotations["data"]["ocr"], local_image_path)

    image = Image.open(image_path)
    NORMALIZED_IMG_LENGTH = 1000
    normalized_img = normalize_image_for_layoutlmv3(image, NORMALIZED_IMG_LENGTH)
    print(bboxes)

    return {
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "image": normalized_img.convert("RGB"),
    }


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list
