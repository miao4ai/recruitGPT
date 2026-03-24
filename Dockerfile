FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_pipeline.txt .
RUN pip install --no-cache-dir -r requirements_pipeline.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY .env.example .env.example

# GCS output dirs will be mounted or written locally then uploaded
RUN mkdir -p data/jds data/resumes data/generated eval

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/run_data_pipeline.py"]
