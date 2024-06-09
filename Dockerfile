# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and pre-trained models
COPY . .

# Expose the port for Cloud Run
EXPOSE 5000

# Set the environment variable for the port
ENV PORT=5000

# Run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "app:app"]
