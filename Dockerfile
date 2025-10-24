FROM python:3.11-alpine

# Install ffmpeg
RUN apk --no-cache add ffmpeg

WORKDIR /app

# Copy and install dependencies
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy the application script
COPY main.py .

# Run the script
CMD ["python", "-u", "main.py"]
