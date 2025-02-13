# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for the Hugging Face token
ENV HF_TOKEN=${HF_TOKEN}

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "whisperx/transcribe.py"]
