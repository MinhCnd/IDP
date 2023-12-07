# IDP

# Serve images for training
\scripts\serve_local_files.sh <path-to-images-folder>

# Run label studio
poetry run label-studio

# Create Train-Test datset from Label Studio annotations
poetry run scripts\build_dataset.py <ls-annotation-file> <output-folder> --s <test-split>

# Run training & inference notebook
poetry run jupyter notebook

# Serve local REST API
poetry run uvicorn idp.main:app --reload

# Build docker image
docker build -t <image_name> .

# Run docker image, mapping port 8080 to container's host
docker run --name <container_name> -d -i -t -p 8080:8080 watercare-idp

# Run tests
poetry run pytest