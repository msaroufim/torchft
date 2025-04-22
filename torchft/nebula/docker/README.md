# Docker-based Testing Environment for Nebula Integration

This directory contains scripts and configuration files for setting up a Docker-based testing environment to validate the Nebula integration with TorchFT for federated learning over residential IPs.

## Prerequisites

- Docker installed on your system
- Nebula binary installed (for certificate generation)
- Basic knowledge of Docker and networking

## Quick Start

1. Make the scripts executable:

```bash
chmod +x setup.sh run_tests.sh cleanup.sh
```

2. Run the setup script to create the test environment:

```bash
./setup.sh
```

This script will:
- Create directories for the test environment
- Generate Nebula certificates
- Create a Docker network
- Build the Docker image
- Copy necessary files

3. Start the Docker containers:

```bash
# Start the lighthouse container
docker run -d --name lighthouse --ip 172.20.0.2 --network nebula-test \
  --cap-add NET_ADMIN --cap-add SYS_MODULE \
  -v ~/nebula-test/lighthouse:/app \
  -v ~/nebula-test/shared:/shared \
  nebula-test

# Start the worker container
docker run -d --name worker --ip 172.20.0.3 --network nebula-test \
  --cap-add NET_ADMIN --cap-add SYS_MODULE \
  -v ~/nebula-test/worker:/app \
  -v ~/nebula-test/shared:/shared \
  nebula-test
```

4. Verify the containers and Nebula setup:

```bash
# Check lighthouse container
docker exec -it lighthouse ip addr show nebula1

# Check worker container
docker exec -it worker ip addr show nebula1
```

5. Run the tests:

```bash
./run_tests.sh
```

6. Clean up when done:

```bash
./cleanup.sh
```

## Manual Testing

If you want to run tests manually or experiment with the setup:

1. Start a shell in one of the containers:

```bash
docker exec -it lighthouse bash
```

2. Run the test script manually:

```bash
cd /shared/torchft
export TORCHFT_USE_NEBULA=1
python3 -m torchft.nebula_test --role coordinator --world-size 2
```

## Troubleshooting

If you run into issues:

1. Check the Nebula connectivity:

```bash
# From lighthouse
docker exec -it lighthouse ping 192.168.100.2

# From worker
docker exec -it worker ping 192.168.100.1
```

2. Check if the status server is running:

```bash
docker exec -it lighthouse netstat -tuln | grep 29501
```

3. Look for errors in the Python output:

```bash
docker exec -it lighthouse bash -c "cd /shared/torchft && export TORCHFT_USE_NEBULA=1 && python3 -m torchft.nebula_test --role coordinator --world-size 2 --verbose"
```

## Files in This Directory

- `Dockerfile`: Defines the Docker image with Nebula and Python
- `start.sh`: Startup script for the containers
- `lighthouse-config.yml`: Nebula configuration for the lighthouse node
- `worker-config.yml`: Nebula configuration for the worker node
- `setup.sh`: Sets up the test environment
- `run_tests.sh`: Runs the federated learning tests
- `cleanup.sh`: Cleans up containers and network