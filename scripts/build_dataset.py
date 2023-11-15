import argparse
import json
from datasets import Dataset, ClassLabel, Sequence
from idp.annotations.annotation_utils import (
    ls_annotations_to_layoutlmv3,
    CLASS_TO_LABEL_MAP,
)

parser = argparse.ArgumentParser(
    "Dataset Builder", "Build your own custom dataset for layout LMv3"
)

parser.add_argument("annotationsFile", help="Input annotation file from Label Studio")
parser.add_argument("output", help="Folder path for saving dataset")
parser.add_argument(
    "--s",
    dest="testSplit",
    help="Percentage of samples to be used for testing, value between 0.0 and 1.0",
)

args = parser.parse_args()

LOCAL_IMAGE_PATH = "datasets\\watercare\\images\\"

with open(args.annotationsFile, "r") as file:

    def sample_generator():
        annotations = json.load(file)
        for index, annotation in enumerate(annotations):
            sample = ls_annotations_to_layoutlmv3(annotation, LOCAL_IMAGE_PATH)
            sample["id"] = str(index)
            if len(sample["tokens"]) != 0:
                yield sample

    dataset = Dataset.from_generator(sample_generator)
    class_names = list(CLASS_TO_LABEL_MAP.values())
    dataset = dataset.cast_column(
        "ner_tags",
        Sequence(ClassLabel(num_classes=len(class_names), names=class_names)),
    )
    if args.testSplit and float(args.testSplit) > 0 and float(args.testSplit) < 1:
        dataset = dataset.train_test_split(test_size=float(args.testSplit))
    dataset.save_to_disk(args.output)
    file.close()
