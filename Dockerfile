# Start from the base Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .dvc config file for authentication
COPY .dvc/config.local /root/.dvc/config.local

# Run the application
CMD ["python", "app.py"]