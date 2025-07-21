# Stage 1: Install dependencies
FROM python:3.12.3-slim AS builder

WORKDIR /app

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt


# Stage 2: Create the final image
FROM python:3.12.3-slim

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy the application source code
COPY . .

# Add the project's root to the Python path to allow imports from src
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# The default command to run when the container starts
# This will start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]