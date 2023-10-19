# import module
from pdf2image import convert_from_path
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    prog="PDF2Image converter", description="Convert pdfs to images"
)

parser.add_argument("input")
parser.add_argument("output")

args = parser.parse_args()

path = Path(args.input)
output = Path(args.output)

if path.is_dir():
    for docPath in path.iterdir():
        images = convert_from_path(docPath, output_folder=output, fmt="png")
else:
    images = convert_from_path(path, output_folder=output, fmt="png")
