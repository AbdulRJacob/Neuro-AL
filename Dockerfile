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
    gcc \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install SWI-Prolog
RUN apt-get update && \
    apt-get install -y swi-prolog && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /neuro_al

VOLUME ["/neuro_al"]

# Copy the requirements file into the container
COPY req.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r req.txt

# Set Python Path
ENV PYTHONPATH="/neuro_al:$PYTHONPATH"

# Copy the rest of the application code into the container
COPY src/ .
RUN mkdir symbolic_modules
COPY symbolic_modules/ symbolic_modules/
RUN mkdir data_structures
COPY data_structures/ data_structures/

# Set the command to run container
CMD ["tail", "-f", "/dev/null"]
