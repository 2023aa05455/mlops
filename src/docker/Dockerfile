# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir flask scikit-learn

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
