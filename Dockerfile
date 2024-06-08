# Use a slim Python base image
FROM python:3.10-slim

# Copy the app.py file into the container
COPY app.py .

# Install any required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Cloud Run
EXPOSE 8080

# Define environment variable for dynamic port assignment
ENV PORT=8080

# Run the application using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
