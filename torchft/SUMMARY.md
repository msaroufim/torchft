# Federated Learning over Residential IPs using Nebula

## Overview

This implementation adds support to TorchFT for conducting federated learning over residential IP networks using Nebula VPN. The solution includes utilities for network discovery, connection management, and robust error handling in challenging network environments.

## Key Components

1. **Network Utilities** (`network_utils.py`):
   - Automatic detection of Nebula VPN interfaces and IPs
   - Functions to check Nebula status and configuration
   - Network retry mechanism for handling unreliable connections
   - Service discovery for finding other federated learning nodes

2. **HTTP Server Improvements** (`http.py`):
   - Enhanced HTTP server factory that adapts to residential network conditions
   - Status server for node discovery and coordination
   - Socket optimizations for NAT traversal and unreliable connections

3. **Test Implementation** (`nebula_test.py`):
   - Complete end-to-end example of federated learning over Nebula
   - Coordinator/worker model for distribution
   - Dynamic node discovery and rank assignment
   - Synthetic dataset for validation

4. **Integration Tests**:
   - Unit tests for network utilities (`network_utils_test.py`)
   - Unit tests for HTTP server utilities (`http_test.py`)

5. **Documentation** (`README_NEBULA.md`):
   - Setup guide for Nebula integration
   - Environment variables reference
   - Troubleshooting tips

## Design Considerations

1. **Non-invasive Integration**:
   - All Nebula support is implemented through environment variables and optional components
   - No changes required to core TorchFT functionality
   - Works with existing PyTorch distributed primitives

2. **Robustness**:
   - Automatic retries for unreliable connections
   - Support for node discovery in dynamic network environments
   - Graceful handling of network interruptions

3. **Extensibility**:
   - The `NebulaCoordinator` class can be extended for more complex orchestration
   - The `StatusServer` provides a foundation for building more advanced coordination services

## Testing Strategy

The implementation includes three testing approaches:

1. **Unit Tests**:
   - Tests for individual components with mocked dependencies
   - Validates core functionality in isolation

2. **Integration Tests**:
   - End-to-end test script that demonstrates the complete workflow
   - Validates coordination between multiple nodes

3. **Manual Validation**:
   - Instructions for setting up and testing on real Nebula networks
   - Diagnostics for troubleshooting network issues

## Usage Example

```python
import torch
import torch.distributed as dist
import torchft
from torchft.network_utils import check_nebula_status, NebulaCoordinator
from torchft.http import StatusServer

# Check Nebula environment
nebula_status = check_nebula_status()
print(f"Nebula status: {nebula_status}")

# Start a status server for discovery
status_server = StatusServer(status_info={"role": "coordinator"})
status_server.start()

# Initialize process group with longer timeout for residential networks
dist.init_process_group(
    backend="gloo",
    timeout=torch.distributed.Store.TIMEOUT_HUNDRED_SECONDS
)

# Create a model with DDP
model = torchft.DistributedDataParallel(MyModel())

# Run training...

# Clean up
dist.destroy_process_group()
status_server.stop()
```

## Future Improvements

1. Add more advanced node failure detection and recovery
2. Implement bandwidth-aware optimization scheduling
3. Add support for encrypted parameter exchange
4. Improve the coordination protocol for large-scale deployments