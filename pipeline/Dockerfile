FROM deepmi/fastsurfer:latest

USER root

# Set the working directory
WORKDIR /app

# Install necessary packages only if they are not already installed
RUN apt-get update && \
    apt-get install -y python3-pyqt5 || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies only if they are not already installed
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt || true

# Copy the source code
COPY src /app/src

USER nonroot

CMD ["/bin/bash"]