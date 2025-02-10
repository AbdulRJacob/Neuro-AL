# Use the official Ubuntu image from Docker Hub
FROM ubuntu:22.04

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a working directory
WORKDIR /neuro_al

# Copy the requirements file into the container
COPY req.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN apt-get install gcc python3-dev
RUN pip3 install --no-cache-dir -r req.txt

# Copy the rest of the application code into the container
COPY src/ .

# Expose port 80 if needed (adjust as necessary)
EXPOSE 80

# Set the command to run your application (adjust as necessary)
CMD ["bash"]

