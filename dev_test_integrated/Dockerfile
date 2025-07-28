# Use a lightweight official Python image
FROM python:3.11-slim-buster

# Set working directory in the container
WORKDIR /app

# Install system dependencies for OCR (Tesseract and Poppler)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code - IMPORTANT: Change this line
COPY main_hugging_face.py main.py # Copy main_hugging_face.py and rename it to main.py inside the container

# Create a directory for temporary invoice files
RUN mkdir -p /app/temp_invoices

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run your FastAPI application with Uvicorn - IMPORTANT: Change this line
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]