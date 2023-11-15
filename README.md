# IDP

# Serve images for training
\scripts\serve_local_files.sh <path-to-images-folder>

# Run label studio
poetry run label-studio

# Create Train-Test datset from Label Studio annotations
poetry run scripts\build_dataset.py <ls-annotation-file> <output-folder> --s <test-split>

# Run training & inference notebook
poetry run jupyter notebook

