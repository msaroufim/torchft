#!/bin/bash
set -e

# Function to check if container exists and is running
check_container() {
    if ! docker ps -q -f name=$1 | grep -q .; then
        echo "Container $1 is not running. Please start it first."
        exit 1
    fi
}

# Check if containers are running
check_container "lighthouse"
check_container "worker"

# Check Nebula connectivity
echo "Testing Nebula connectivity..."
docker exec -it lighthouse ping -c 2 192.168.100.2
docker exec -it worker ping -c 2 192.168.100.1

echo "Nebula connectivity verified!"

# Run the tests
echo "Starting coordinator on lighthouse..."
docker exec -it lighthouse bash -c "cd /shared/torchft && export TORCHFT_USE_NEBULA=1 && python3 -m torchft.nebula_test --role coordinator --world-size 2" &
COORDINATOR_PID=$!

# Give coordinator time to start
sleep 5

echo "Starting worker..."
docker exec -it worker bash -c "cd /shared/torchft && export TORCHFT_USE_NEBULA=1 && python3 -m torchft.nebula_test --role worker --master 192.168.100.1"

# Wait for coordinator to finish
wait $COORDINATOR_PID

echo "Tests completed!"