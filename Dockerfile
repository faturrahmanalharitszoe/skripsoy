# Use a slim Python base image
FROM python:3.10-slim

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and pre-trained models
COPY app.py crop.pt ocr.pt /app/

# Set the working directory
WORKDIR /app

# Expose the port for Cloud Run
EXPOSE 8080

# Define environment variable for dynamic port assignment
ENV PORT=8080

# Run the Flask application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
