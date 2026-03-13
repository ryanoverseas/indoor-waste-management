# Using the full Python image instead of -slim to avoid missing repository issues on some VPS providers
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies with a retry/ignore pattern for "Show and Tell" deployment
RUN apt-get update --fix-missing && \
    (apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git || true) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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
