# Federated Learning over Residential IPs with Nebula

This guide explains how to use TorchFT with [Nebula](https://github.com/slackhq/nebula) to enable federated learning in non-uniform network environments, especially over residential IP addresses.

## Overview

Nebula is a scalable overlay networking tool that creates a virtual network connecting computers across the internet, even behind NAT and firewalls. This makes it ideal for federated learning across residential networks or other challenging network environments.

TorchFT provides built-in support for Nebula networking, allowing you to:

1. Run distributed PyTorch training across residential networks
2. Automatically discover other nodes on the Nebula network
3. Handle connection retries and network instability
4. Coordinate federated learning jobs across multiple machines

## Setup Guide

### 1. Install and Configure Nebula

First, you need to set up Nebula on all machines that will participate in the federated learning job. Follow the [official Nebula installation guide](https://nebula.defined.net/docs/guides/quick-start).

The basic steps are:

1. **Set up a lighthouse node** - This should be a machine with a public IP address that acts as a discovery node
2. **Generate certificates** for all nodes using `nebula-cert`
3. **Configure each node** with the appropriate `config.yml` file
4. **Start Nebula** on all nodes

### 2. Environment Variables

TorchFT uses the following environment variables for Nebula integration:

```bash
# Required: Enable Nebula mode
export TORCHFT_USE_NEBULA=1

# Optional: Specify Nebula interface if auto-detection doesn't work
export TORCHFT_NEBULA_INTERFACE=nebula1

# Optional: Manually specify external/Nebula IP if auto-detection doesn't work
export TORCHFT_EXTERNAL_IP=192.168.100.1

# Optional: Comma-separated list of known Nebula peers
export TORCHFT_NEBULA_PEERS=192.168.100.2,192.168.100.3

# Optional: Connection retry settings
export TORCHFT_CONNECT_RETRY=10      # Number of retries
export TORCHFT_CONNECT_TIMEOUT=5.0   # Seconds between retries
```

### 3. Create a Federated Learning Job

You can use the included `nebula_test.py` script to test your Nebula setup:

**On the coordinator node (usually the lighthouse):**
```bash
python -m torchft.nebula_test --role coordinator --world-size 3
```

**On worker nodes:**
```bash
python -m torchft.nebula_test --role worker --master 192.168.100.1
```

## Using Nebula in Your Code

To use Nebula in your own code:

```python
import torchft
from torchft.network_utils import check_nebula_status, NebulaCoordinator
from torchft.http import StatusServer

# Check Nebula status
nebula_status = check_nebula_status()
print(f"Nebula status: {nebula_status}")

# Start a status server for discovery
status_server = StatusServer(status_info={"role": "coordinator"})
status_server.start()

# Create a coordinator for discovery
coordinator = NebulaCoordinator()

# Discover other nodes
services = coordinator.discover_services()
print(f"Discovered services: {services}")

# Use standard PyTorch distributed with better timeouts
import torch.distributed as dist
dist.init_process_group(
    backend="gloo",
    timeout=torch.distributed.Store.TIMEOUT_HUNDRED_SECONDS
)

# Clean up
status_server.stop()
```

## Troubleshooting

### Connection Issues

If nodes can't connect to each other:

1. **Check Nebula status** with `sudo nebula-cert verify -ca-crt lighthouse.crt client.crt`
2. **Verify Nebula interface** is active with `ip addr show`
3. **Ping other nodes** using their Nebula IPs
4. **Check firewall rules** to ensure ports are open

### Performance Considerations

- Use the `NetworkRetryHandler` for operations that might fail on unreliable networks
- Set longer timeouts for distributed operations
- Consider increasing batch size to reduce communication frequency
- For large models, use parameter sharding to reduce per-node memory requirements