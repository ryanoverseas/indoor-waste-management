# Use an official Python runtime as a parent image
# Using 3.11-slim to keep the image size manageable
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Note: This installs the CPU version of Torch by default. 
# If your Hostinger KVM has a GPU, we would need to adjust the index-url.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Create a folder for results if it doesn't exist
RUN mkdir -p runs/detect

# The container will stay alive so you can run training manually via Dokploy Terminal
# Or you can change this to start a specific model training
CMD ["tail", "-f", "/dev/null"]
