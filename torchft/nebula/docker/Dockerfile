FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    wget \
    iproute2 \
    net-tools \
    iputils-ping \
    python3 \
    python3-pip \
    netcat \
    curl \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Nebula
RUN wget https://github.com/slackhq/nebula/releases/download/v1.7.2/nebula-linux-amd64.tar.gz \
    && tar -xzf nebula-linux-amd64.tar.gz \
    && mv nebula /usr/local/bin/ \
    && mv nebula-cert /usr/local/bin/ \
    && rm nebula-linux-amd64.tar.gz

# Install Python packages
RUN pip3 install torch torchvision torchaudio requests pyyaml

WORKDIR /app

# Add a startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]