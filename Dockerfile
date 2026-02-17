
# PaperRAG Production Dockerfile

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     git     && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Create virtual environment
RUN uv venv --python 3.10

# Activate and install dependencies
RUN . .venv/bin/activate &&     uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["paperrag", "index"]
