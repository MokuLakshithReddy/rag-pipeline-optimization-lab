FROM python:3.12-slim

WORKDIR /app

# Install essential system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency resolution
RUN pip install --no-cache-dir uv

# Copy project specification and complete source tree before installation
COPY . .

# Install package in editable mode with all dependencies
RUN uv pip install --system -e .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
