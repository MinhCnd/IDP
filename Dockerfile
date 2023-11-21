FROM python:3.10-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock /app/

# Disable virtualenv creation and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Copy the rest of the application files to the container
COPY . /app

# Install tesseract ocr
RUN apt-get update && apt-get -y install tesseract-ocr

RUN pip install torch

# Install pdf2image dependency
RUN apt-get -y install poppler-utils

# Inject model path into env
RUN echo "MODEL_PATH=$INPUT_MODEL_PATH" >> .env

EXPOSE 8080

# # Run the application on port 8080
CMD ["uvicorn", "idp.main:app", "--host", "0.0.0.0", "--port", "8080"]