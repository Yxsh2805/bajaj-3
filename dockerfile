FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies including setuptools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install setuptools first
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
