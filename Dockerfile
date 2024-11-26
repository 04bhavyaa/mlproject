# Start from the base Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Copy the .dvc config file for authentication
COPY .dvc/config /root/.dvc/config

# Copy the .dvc cache directory for data versioning
COPY .dvc/cache /root/.dvc/cache

# Pull the data using DVC
RUN dvc pull

# Run the application
CMD ["python", "app.py"]