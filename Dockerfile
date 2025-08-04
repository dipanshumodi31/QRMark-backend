# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by OpenCV and other libs
# libzbar0 is for pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    # You can add other system dependencies here if needed in the future
    # e.g., libjpeg-dev zlib1g-dev for Pillow, or libgl1-mesa-glx for some OpenCV setups
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt # <--- CHANGED THIS LINE

# Copy the rest of your application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
# Railway typically injects $PORT, so use that
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# If Railway requires explicit $PORT (check Railway's docs/logs if it doesn't run):
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]
