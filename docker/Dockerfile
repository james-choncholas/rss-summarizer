# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script and the processed IDs file
COPY ../main.py .
# Copying the initial state. For persistent state, mount this file as a volume:
# docker run -v $(pwd)/processed_article_ids.json:/app/processed_article_ids.json ...
COPY ../processed_article_ids.json .

# --- IMPORTANT --- 
# The .env file containing secrets like OPENAI_API_KEY should NOT be copied directly.
# Provide the API key via Docker environment variables (-e) or secrets management.
# Example: docker run -e OPENAI_API_KEY=your_key ...

# Define environment variable defaults (can be overridden at runtime)
ENV REQUEST_DELAY_SECONDS=1
ENV REQUEST_TIMEOUT_SECONDS=15
ENV CHECK_INTERVAL_MINUTES=30
ENV SUMMARY_TIME="17:00"
ENV MODEL_NAME="gpt-3.5-turbo"
ENV TEMPERATURE=0.3

# Command to run the application
# The feed URL must be provided as an argument when running the container.
# Example: docker run your-image-name "http://example.com/rss.xml"
ENTRYPOINT ["python", "main.py"]
# Default command if no arguments are provided to ENTRYPOINT (optional, shows usage)
CMD ["--help"] 