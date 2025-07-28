# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the application files
COPY extraction.py .
COPY models/ ./models/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entry point to automatically process input directory
CMD ["python", "extraction.py", "/app/input", "/app/output"]
