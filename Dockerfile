# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Expose port Flask will run on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
