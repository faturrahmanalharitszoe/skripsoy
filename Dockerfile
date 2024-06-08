# Use an official lightweight Python image.
# 3.8-slim is chosen for its balance between size and utility.
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

# Run the application:
# Use gunicorn as the entry point for running the application. Gunicorn is a WSGI HTTP Server for UNIX.
# It's a pre-fork worker model ported from Ruby's Unicorn project.
# The "-b :$PORT" option binds gunicorn to the port defined by the PORT environment variable.
CMD ["python", "app.py"]

