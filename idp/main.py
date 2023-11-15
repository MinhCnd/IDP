from transformers import AutoModelForTokenClassification
from fastapi import FastAPI, UploadFile
from pdf2image import convert_from_bytes
from transformers import AutoProcessor
import pytesseract
import torch
from idp.annotations.bbox_utils import (
    unnormalize_box,
    normalize_box,
    merge_box_extremes,
)
from idp.annotations.annotation_utils import (
    Classes,
    CLASS_TO_LABEL_MAP,
    LABEL_STR_TO_CLASS_MAP,
)

app = FastAPI()

# Instatiate model & processor
MODEL_PATH = "C:/Projects/IDP/watercare/model_output/23_11_03_03/checkpoint-150"
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False)


def is_box_a_within_box_b(box_a, box_b):
    left_a, top_a, right_a, bottom_a = box_a
    left_b, top_b, right_b, bottom_b = box_b

    # Check if Box B contains box A
    return (
        left_b <= left_a
        and top_b <= top_a
        and right_b >= right_a
        and bottom_b >= bottom_a
    )


def get_text_box_pairs(image):
    img_width, img_height = image.size
    tesseract_output = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT
    )
    boxes = []
    texts = []
    for i, level_idx in enumerate(tesseract_output["level"]):
        if level_idx == 5:
            bbox = [
                tesseract_output["left"][i],
                tesseract_output["top"][i],
                tesseract_output["left"][i] + tesseract_output["width"][i],
                tesseract_output["top"][i] + tesseract_output["height"][i],
            ]
            if not tesseract_output["text"][i].strip():
                continue
            bbox = normalize_box(bbox, img_width, img_height)
            texts.append(tesseract_output["text"][i])
            boxes.append(bbox)
    return (texts, boxes)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
        contents = file.file.read()
        images = convert_from_bytes(contents)

        # Extract texts using OCR
        pages = len(images)
        text_box_pairs = [get_text_box_pairs(image) for image in images]
        texts = [pair[0] for pair in text_box_pairs]
        boxes = [pair[1] for pair in text_box_pairs]
        page_indexes = [[index] * len(text_arr) for index, text_arr in enumerate(texts)]

        # Assume all pages have the same size
        img_width, img_height = images[0].size

        def text_box_relevant(text_box_page):
            """HACK remove texts & boxes outside of Labelling area"""
            text, box, page = text_box_page

            if page == 0:
                outer_box = normalize_box(
                    (1298, 828, 1536, 1550), img_width, img_height
                )
                return is_box_a_within_box_b(box, outer_box)
            elif page == 1:
                outer_box = normalize_box((139, 119, 827, 1173), img_width, img_height)
                return is_box_a_within_box_b(box, outer_box)
            else:
                return False

        out_texts = []
        out_boxes = []

        for page in range(pages):
            filtered_list = list(
                filter(
                    text_box_relevant,
                    list(zip(texts[page], boxes[page], page_indexes[page])),
                )
            )
            results = [
                [text for text, box, page in filtered_list],
                [box for text, box, page in filtered_list],
            ]
            temp_texts, temp_boxes = results
            out_texts.append(temp_texts)
            out_boxes.append(temp_boxes)

        # Encode input
        encoding = processor(
            images=images,
            text=out_texts,
            boxes=out_boxes,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Perform inferece
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(-1).tolist()

            token_boxes = encoding.bbox.tolist()
            pages_input_ids = encoding["input_ids"].tolist()
            PAD_ID = 1
            # First token is a special token, ignore
            pad_indexes = [
                page_input_ids.index(PAD_ID)
                for page_input_ids in encoding["input_ids"].tolist()
            ]
            true_predictions = [
                [
                    model.config.id2label[pred]
                    for pred in page_preds[1 : pad_indexes[page_index]]
                ]
                for page_index, page_preds in enumerate(predictions)
            ]
            true_boxes = [
                [
                    unnormalize_box(box, img_width, img_height)
                    for box in page_token_boxes[1 : pad_indexes[page_index]]
                ]
                for page_index, page_token_boxes in enumerate(token_boxes)
            ]
            true_texts = [
                [
                    processor.tokenizer.decode([value])
                    for index, value in enumerate(
                        page_input_ids[1 : pad_indexes[page_index]]
                    )
                ]
                for page_index, page_input_ids in enumerate(pages_input_ids)
            ]

            class_to_label_str_map = {v: k for k, v in LABEL_STR_TO_CLASS_MAP.items()}
            output = [
                {
                    class_to_label_str_map[key]: {"text": "", "box": []}
                    for key, value in CLASS_TO_LABEL_MAP.items()
                    if key != Classes.OTHER
                }
                for page in range(pages)
            ]
            output = [
                {
                    class_to_label_str_map[key]: {"text": ""}
                    for key, value in CLASS_TO_LABEL_MAP.items()
                    if key != Classes.OTHER
                }
                for page in range(pages)
            ]

            for page_indx in range(pages):
                for key, value in CLASS_TO_LABEL_MAP.items():
                    if key == Classes.OTHER:
                        continue

                    output[page_indx][class_to_label_str_map[key]]["text"] = "".join(
                        [
                            text
                            for text, prediction, box in zip(
                                true_texts[page_indx],
                                true_predictions[page_indx],
                                true_boxes[page_indx],
                            )
                            if (prediction == value and box != [0, 0, 0, 0])
                        ]
                    )
                    output[page_indx][class_to_label_str_map[key]][
                        "box"
                    ] = merge_box_extremes(
                        [
                            box
                            for text, prediction, box in zip(
                                true_texts[page_indx],
                                true_predictions[page_indx],
                                true_boxes[page_indx],
                            )
                            if (prediction == value and box != [0, 0, 0, 0])
                        ]
                    )

            # #trim empty outputs
            def item_not_empty(item):
                return len(item[1]["text"]) != 0

            filtered_output = [
                dict(filter(item_not_empty, page_output.items()))
                for page_output in output
            ]

            return filtered_output
    except Exception:
        return {"message": "There was an error reading the file"}
    finally:
        file.file.close()
