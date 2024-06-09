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
ENV FLASK_APP=app.py

# Run the Flask application with Gunicorn
CMD ["flask", "run", "--host=0.0.0.0"]
