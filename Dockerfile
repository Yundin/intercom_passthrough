FROM mwader/static-ffmpeg:latest as ffmpeg-bin

FROM python:3-alpine

# Install ffmpeg
COPY --from=ffmpeg-bin /ffmpeg /usr/local/bin/
COPY --from=ffmpeg-bin /ffprobe /usr/local/bin/

WORKDIR /app

# Copy and install dependencies
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy the application script
COPY main.py .

# Run the script
CMD ["python", "-u", "main.py"]
