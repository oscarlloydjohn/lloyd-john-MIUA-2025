FROM deepmi/fastsurfer:latest

# Set the working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && \
    apt-get install -y python3-pyqt5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the source code
COPY src /app/src

CMD ["python3", "src/pipeline.py"]