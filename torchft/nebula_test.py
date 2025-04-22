#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for Nebula networking support in TorchFT
---------------------------------------------

This script tests and demonstrates federated learning over residential IPs
using Nebula VPN. Run this script on multiple machines connected via Nebula
to test the functionality.

Usage:
    # Set up environment
    export TORCHFT_USE_NEBULA=1  # Enable Nebula mode
    export TORCHFT_NEBULA_PEERS="192.168.100.1,192.168.100.2"  # Known peers (optional)
    
    # Run as coordinator (lighthouse)
    python -m torchft.nebula_test --role coordinator
    
    # Run as worker on other machines
    python -m torchft.nebula_test --role worker --master <coordinator_ip>
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchft
from torchft.network_utils import (
    check_nebula_status, 
    get_external_ip,
    is_nebula_enabled,
    get_nebula_ip,
    NetworkRetryHandler,
    NebulaCoordinator,
)
from torchft.http import StatusServer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simple synthetic dataset for testing
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100):
        self.data = torch.randn(num_samples, 10)
        self.targets = torch.randint(0, 2, (num_samples,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def setup_nebula_environment():
    """Check and set up the Nebula environment."""
    # Check if Nebula is properly configured
    status = check_nebula_status()
    
    if not status["enabled"]:
        logger.error("Nebula is not enabled. Please set TORCHFT_USE_NEBULA=1")
        sys.exit(1)
        
    if not status.get("interface"):
        logger.warning("Nebula interface not detected. Check Nebula installation.")
    
    if not status.get("ip"):
        logger.warning("Couldn't detect Nebula IP address. Discovery may not work.")
        
    logger.info(f"Nebula status: {json.dumps(status, indent=2)}")
    
    return status

def run_coordinator(args):
    """Run in coordinator mode."""
    logger.info("Starting in coordinator mode...")
    
    # Set up Nebula environment
    nebula_status = setup_nebula_environment()
    
    # Start status server
    status_server = StatusServer(
        port=args.port,
        status_info={
            "role": "coordinator",
            "world_size": args.world_size,
            "job_id": args.job_id,
        }
    )
    status_server.start()
    
    # Create a coordinator for service discovery
    coordinator = NebulaCoordinator()
    
    # Wait for workers to connect
    connected_workers = []
    logger.info(f"Waiting for {args.world_size-1} workers to connect...")
    
    start_time = time.time()
    while len(connected_workers) < args.world_size - 1:
        if args.timeout > 0 and time.time() - start_time > args.timeout:
            logger.error(f"Timeout waiting for workers. Connected: {len(connected_workers)}")
            sys.exit(1)
            
        # Discover workers using the coordinator
        services = coordinator.discover_services(port=args.port)
        
        # Filter for worker nodes with our job ID
        for ip, url in list(services.items()):
            if ip not in [worker["ip"] for worker in connected_workers]:
                try:
                    # Check if it's a worker for our job
                    response = requests.get(f"{url}/status", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if (data.get("role") == "worker" and 
                            data.get("job_id") == args.job_id):
                            
                            # Add to connected workers
                            connected_workers.append({
                                "ip": ip,
                                "url": url,
                                "rank": len(connected_workers) + 1
                            })
                            
                            logger.info(f"Worker connected: {ip} (rank {len(connected_workers)})")
                except Exception as e:
                    logger.warning(f"Error checking worker at {url}: {e}")
        
        # Update status with connected workers
        status_server.update_status({
            "connected_workers": len(connected_workers),
            "workers": connected_workers
        })
        
        time.sleep(2)
    
    # All workers connected, start the job
    logger.info(f"All {args.world_size-1} workers connected, starting the job...")
    
    # Set up process group
    os.environ["MASTER_ADDR"] = nebula_status["ip"]
    os.environ["MASTER_PORT"] = str(args.dist_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.world_size)
    
    # Initialize process group with better timeout for residential connections
    dist.init_process_group(
        backend="gloo", 
        timeout=torch.distributed.Store.TIMEOUT_HUNDRED_SECONDS
    )
    
    # Update status
    status_server.update_status({"dist_status": "initialized"})
    
    # Simple training example
    model = SimpleModel()
    model = torchft.DistributedDataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dataset and dataloader
    dataset = SyntheticDataset()
    sampler = torchft.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    # Run a few epochs
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.6f}")
        
        # Synchronize at the end of each epoch
        dist.barrier()
        
        logger.info(f"Completed epoch {epoch}")
        
        # Update status
        status_server.update_status({
            "training_status": f"Completed epoch {epoch+1}/{args.epochs}"
        })
    
    # Clean up
    dist.destroy_process_group()
    status_server.update_status({"status": "complete"})
    
    logger.info("Coordinator job complete")
    
    # Keep server running for a while to let workers get the final status
    time.sleep(10)
    status_server.stop()

def run_worker(args):
    """Run in worker mode."""
    logger.info("Starting in worker mode...")
    
    # Set up Nebula environment
    nebula_status = setup_nebula_environment()
    
    if not args.master:
        logger.error("Master address (--master) is required for worker mode")
        sys.exit(1)
    
    # Start status server
    status_server = StatusServer(
        port=args.port,
        status_info={
            "role": "worker",
            "job_id": args.job_id,
        }
    )
    status_server.start()
    
    # Create retry handler for connecting to coordinator
    retry_handler = NetworkRetryHandler()
    
    # Connect to coordinator and get rank
    def get_coordinator_info():
        import requests
        url = f"http://{args.master}:{args.port}/status"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to connect to coordinator: HTTP {response.status_code}")
        return response.json()
    
    coordinator_info = retry_handler.execute_with_retry(get_coordinator_info)
    
    if coordinator_info.get("role") != "coordinator":
        logger.error("The specified master is not a coordinator")
        sys.exit(1)
    
    # Wait for our worker to be registered with the coordinator
    my_ip = nebula_status["ip"]
    my_rank = None
    
    logger.info(f"Waiting to be registered with coordinator...")
    
    start_time = time.time()
    while my_rank is None:
        if args.timeout > 0 and time.time() - start_time > args.timeout:
            logger.error("Timeout waiting to be registered with coordinator")
            sys.exit(1)
            
        try:
            coordinator_info = retry_handler.execute_with_retry(get_coordinator_info)
            workers = coordinator_info.get("workers", [])
            
            for worker in workers:
                if worker["ip"] == my_ip:
                    my_rank = worker["rank"]
                    logger.info(f"Registered with coordinator as rank {my_rank}")
                    break
                    
        except Exception as e:
            logger.warning(f"Error getting coordinator info: {e}")
            
        time.sleep(5)
    
    # Update status
    status_server.update_status({"rank": my_rank})
    
    # Wait for all workers to connect before initializing process group
    world_size = coordinator_info.get("world_size", 0)
    
    while coordinator_info.get("connected_workers", 0) < world_size - 1:
        logger.info("Waiting for all workers to connect...")
        time.sleep(5)
        coordinator_info = retry_handler.execute_with_retry(get_coordinator_info)
    
    # Set up process group
    os.environ["MASTER_ADDR"] = args.master
    os.environ["MASTER_PORT"] = str(args.dist_port)
    os.environ["RANK"] = str(my_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Initialize process group with better timeout for residential connections
    dist.init_process_group(
        backend="gloo", 
        timeout=torch.distributed.Store.TIMEOUT_HUNDRED_SECONDS
    )
    
    # Update status
    status_server.update_status({"dist_status": "initialized"})
    
    # Simple training example
    model = SimpleModel()
    model = torchft.DistributedDataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dataset and dataloader
    dataset = SyntheticDataset()
    sampler = torchft.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    # Run training - will be synchronized by coordinator
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.6f}")
        
        # Synchronize at the end of each epoch
        dist.barrier()
        
        logger.info(f"Completed epoch {epoch}")
        
        # Update status
        status_server.update_status({
            "training_status": f"Completed epoch {epoch+1}/{args.epochs}"
        })
    
    # Clean up
    dist.destroy_process_group()
    status_server.update_status({"status": "complete"})
    
    logger.info("Worker job complete")
    
    # Keep server running for a while to let coordinator get the final status
    time.sleep(10)
    status_server.stop()

def main():
    parser = argparse.ArgumentParser(description="Test Nebula networking support")
    parser.add_argument("--role", choices=["coordinator", "worker"], required=True,
                        help="Role of this node")
    parser.add_argument("--master", type=str, default=None,
                        help="Master (coordinator) Nebula IP address (required for worker)")
    parser.add_argument("--port", type=int, default=29501,
                        help="Port for status server")
    parser.add_argument("--dist-port", type=int, default=29500,
                        help="Port for distributed training")
    parser.add_argument("--world-size", type=int, default=2,
                        help="Total number of nodes (default: 2)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--job-id", type=str, default="nebula-test",
                        help="Unique ID for this job")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds for waiting (0 for no timeout)")
    
    args = parser.parse_args()
    
    # Import here to avoid import issues
    import requests
    
    # Run in the appropriate mode
    if args.role == "coordinator":
        run_coordinator(args)
    else:
        run_worker(args)

if __name__ == "__main__":
    main()