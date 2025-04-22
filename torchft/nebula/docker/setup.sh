#!/bin/bash
set -e

# Create directories
mkdir -p ~/nebula-test/{lighthouse,worker,shared}
cd ~/nebula-test

# Copy files from the repo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cp "$SCRIPT_DIR/Dockerfile" .
cp "$SCRIPT_DIR/start.sh" .
cp "$SCRIPT_DIR/lighthouse-config.yml" lighthouse/config.yml
cp "$SCRIPT_DIR/worker-config.yml" worker/config.yml

# Generate Nebula certificates
echo "Generating Nebula certificates..."
nebula-cert ca -name "FederatedLearning CA"

# Generate certificates for lighthouse and worker
nebula-cert sign -name "lighthouse" -ip "192.168.100.1/24" -groups "servers"
nebula-cert sign -name "worker" -ip "192.168.100.2/24" -groups "workers"

# Move certificates to their respective folders
cp ca.crt lighthouse/ && cp ca.crt worker/
cp lighthouse.crt lighthouse.key lighthouse/
cp worker.crt worker.key worker/

# Create Docker network
echo "Creating Docker network..."
docker network create --subnet=172.20.0.0/16 nebula-test || true

# Build Docker image
echo "Building Docker image..."
docker build -t nebula-test .

# Copy TorchFT code to shared directory
echo "Copying TorchFT code..."
cp -r "$SCRIPT_DIR/../.." shared/torchft

echo "Setup complete! You can now start the containers with:"
echo "docker run -d --name lighthouse --ip 172.20.0.2 --network nebula-test --cap-add NET_ADMIN --cap-add SYS_MODULE -v $(pwd)/lighthouse:/app -v $(pwd)/shared:/shared nebula-test"
echo "docker run -d --name worker --ip 172.20.0.3 --network nebula-test --cap-add NET_ADMIN --cap-add SYS_MODULE -v $(pwd)/worker:/app -v $(pwd)/shared:/shared nebula-test"