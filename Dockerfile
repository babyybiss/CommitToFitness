# Use a compatible base image for arm64
FROM --platform=linux/arm64 python:3.11-slim as base

# Stage 1: Install Dependencies
FROM base as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY pyproject.toml /tmp/

# Export requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Production Container
FROM python:3.11.9

# Install necessary system dependencies
#RUN apt-get update && apt-get install -y \
#    python3-opencv

# Set working directory
WORKDIR /COMMITTOFITNESS

# Copy requirements.txt from previous stage
COPY --from=requirements-stage /tmp/requirements.txt /requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

# Copy application code
COPY . /COMMITTOFITNESS

# Command to run the application
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
