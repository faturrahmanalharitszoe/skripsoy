# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 0 --timeout 0 app:app