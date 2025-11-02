# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Flask port
EXPOSE 8080

# Command to run your app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
