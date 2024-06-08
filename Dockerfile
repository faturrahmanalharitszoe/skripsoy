# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for dynamic port assignment by Cloud Run
ENV PORT=8080

# Run the application using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
