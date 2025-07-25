FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/app
ENV PADDLEOCR_HOME=/app/.paddleocr

WORKDIR /app

# Create writable OCR model directory and change ownership
RUN mkdir -p /app/.paddleocr && \
    adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

# Update package list and install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        libglib2.0-dev \
        libcairo2 \
        libpango1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        python3-tk \
        ghostscript \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Switch to non-root user (required by Hugging Face Spaces)
USER appuser

# Expose port (adjust based on what you're running)
EXPOSE 3000

# Start the app
CMD ["python", "app.py"]
