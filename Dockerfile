# Use an official Python runtime as a parent image
FROM python:3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY . .

# Command to run the application
# The feed URL must be provided as an argument when running the container.
# Example: docker run your-image-name "http://example.com/rss.xml"
ENTRYPOINT ["python", "main.py"]