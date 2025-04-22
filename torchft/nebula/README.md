# Federated Learning over Residential IPs with Nebula

This directory contains tools and documentation for integrating [Nebula](https://github.com/slackhq/nebula) VPN with TorchFT to enable federated learning over residential IP networks.

## Overview

Nebula is a scalable overlay networking tool that creates a virtual network connecting computers across the internet, even behind NAT and firewalls. This makes it ideal for federated learning across residential networks.

## Contents

- [`docker/`](docker/) - Docker-based testing environment for validating the integration
- [`README_NEBULA.md`](../README_NEBULA.md) - Detailed guide to using Nebula with TorchFT

## Quick Start

To test Nebula integration with TorchFT:

1. Install Nebula on your system (required for certificate generation):
   ```bash
   # On Ubuntu/Debian
   wget https://github.com/slackhq/nebula/releases/download/v1.7.2/nebula-linux-amd64.tar.gz
   tar -xzf nebula-linux-amd64.tar.gz
   sudo mv nebula nebula-cert /usr/local/bin/
   
   # On macOS
   brew install nebula
   ```

2. Use the Docker-based testing environment:
   ```bash
   cd docker
   chmod +x setup.sh run_tests.sh cleanup.sh
   ./setup.sh
   
   # Follow the instructions printed by the setup script
   ```

3. For more details, see the [Docker README](docker/README.md).

## Integration with Real Nebula Networks

For instructions on integrating with real Nebula networks across residential IPs:

1. Set up Nebula on all participating machines following the [official documentation](https://nebula.defined.net/docs/guides/quick-start/)

2. Configure TorchFT to use Nebula:
   ```bash
   export TORCHFT_USE_NEBULA=1
   export TORCHFT_NEBULA_INTERFACE=nebula1  # Optional, if auto-detection fails
   ```

3. Run the test script:
   ```bash
   # On coordinator (lighthouse)
   python -m torchft.nebula_test --role coordinator --world-size N
   
   # On workers
   python -m torchft.nebula_test --role worker --master <lighthouse-nebula-ip>
   ```

For full details, see the [Nebula integration guide](../README_NEBULA.md).