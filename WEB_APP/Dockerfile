# Use Python 3.10 slim image (smaller than full python image)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required (e.g., libgl1 for torchvision/opencv, build-essential)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (7860 is required for Hugging Face Spaces)
EXPOSE 7860

# Command to run the application using Gunicorn
# Using 1 worker and multiple threads due to high memory usage of PyTorch
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app"]
