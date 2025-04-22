# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Network Utilities for Non-Uniform Networks
=========================================

This module provides utilities for handling non-uniform networks like Nebula and residential IPs.
"""

import logging
import os
import socket
import subprocess
import requests
import json
from typing import List, Optional, Dict, Tuple, Any, Set

logger: logging.Logger = logging.getLogger(__name__)

def get_external_ip() -> str:
    """
    Attempts to determine the external IP address of this machine.
    
    First checks if TORCHFT_EXTERNAL_IP environment variable is set, otherwise
    tries to determine it using a set of external services. 
    
    Returns:
        External IP address as string
    """
    # First check environment variable
    if "TORCHFT_EXTERNAL_IP" in os.environ:
        return os.environ["TORCHFT_EXTERNAL_IP"]
    
    # Check if we should use Nebula IP
    if is_nebula_enabled():
        nebula_ip = get_nebula_ip()
        if nebula_ip:
            logger.info(f"Using Nebula IP address: {nebula_ip}")
            return nebula_ip
    
    # Try to get from external service
    ip_services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://checkip.amazonaws.com",
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service, timeout=5)
            if response.status_code == 200:
                ip = response.text.strip()
                logger.info(f"Determined external IP to be {ip} using {service}")
                return ip
        except Exception as e:
            logger.warning(f"Failed to determine external IP using {service}: {e}")
    
    # Fall back to local hostname
    logger.warning("Could not determine external IP, falling back to hostname")
    return socket.gethostname()

def is_nebula_enabled() -> bool:
    """
    Checks if Nebula networking is enabled by environment variable.
    
    Returns:
        True if Nebula is enabled, False otherwise
    """
    return os.environ.get("TORCHFT_USE_NEBULA", "").lower() in ("1", "true", "yes")

def get_nebula_interface() -> Optional[str]:
    """
    Gets the Nebula network interface name, if available.
    
    Returns:
        Interface name or None if not available
    """
    interface = os.environ.get("TORCHFT_NEBULA_INTERFACE")
    if interface:
        return interface
    
    # Common Nebula interface names to check
    potential_interfaces = ["nebula1", "nebula", "tun0"]
    
    try:
        # Try to detect interface from running 'ip addr' command
        proc = subprocess.run(["ip", "addr"], capture_output=True, text=True)
        if proc.returncode == 0:
            for iface in potential_interfaces:
                if iface in proc.stdout:
                    logger.info(f"Detected Nebula interface: {iface}")
                    return iface
    except Exception as e:
        logger.warning(f"Failed to detect Nebula interface: {e}")
    
    return None

def get_nebula_ip() -> Optional[str]:
    """
    Gets the Nebula IP address for this node.
    
    Returns:
        Nebula IP address or None if not available
    """
    interface = get_nebula_interface()
    if not interface:
        return None
    
    try:
        # First try the 'ip addr' command which works on Linux
        proc = subprocess.run(
            ["ip", "addr", "show", interface], 
            capture_output=True, 
            text=True
        )
        
        if proc.returncode == 0:
            import re
            # Look for IPv4 address pattern in the output
            match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', proc.stdout)
            if match:
                return match.group(1)
    except FileNotFoundError:
        # If 'ip' command is not available (e.g., on macOS)
        try:
            proc = subprocess.run(
                ["ifconfig", interface],
                capture_output=True,
                text=True
            )
            if proc.returncode == 0:
                import re
                # Look for IPv4 address pattern in the output
                match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', proc.stdout)
                if match:
                    return match.group(1)
        except FileNotFoundError:
            pass
    
    logger.warning(f"Could not determine Nebula IP address for interface {interface}")
    return None

def get_nebula_lighthouse_ips() -> List[str]:
    """
    Gets the list of Nebula lighthouse IPs from configuration.
    
    Returns:
        List of lighthouse IP addresses
    """
    lighthouse_ips = os.environ.get("TORCHFT_NEBULA_LIGHTHOUSES", "")
    if lighthouse_ips:
        return [ip.strip() for ip in lighthouse_ips.split(",")]
    
    # Try to parse from nebula config if available
    config_path = os.environ.get("TORCHFT_NEBULA_CONFIG", "/etc/nebula/config.yml")
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        lighthouses = config.get("static_host_map", {})
        ips = []
        for entries in lighthouses.values():
            ips.extend([entry.split(":")[0] for entry in entries])
        
        return ips
    except Exception as e:
        logger.warning(f"Failed to parse Nebula config for lighthouses: {e}")
    
    return []

def get_nebula_hosts() -> Dict[str, str]:
    """
    Gets a mapping of Nebula IPs to hostnames, if available.
    
    Returns:
        Dictionary mapping Nebula IPs to hostnames
    """
    hosts_file = os.environ.get("TORCHFT_NEBULA_HOSTS_FILE", "")
    if not hosts_file or not os.path.exists(hosts_file):
        return {}
    
    hosts = {}
    try:
        with open(hosts_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ip, hostname = parts[0], parts[1]
                        hosts[ip] = hostname
    except Exception as e:
        logger.warning(f"Failed to parse Nebula hosts file: {e}")
    
    return hosts

def check_nebula_status() -> Dict[str, Any]:
    """
    Checks the status of the Nebula VPN connection.
    
    Returns:
        Dictionary with status information
    """
    if not is_nebula_enabled():
        return {"enabled": False}
    
    result = {
        "enabled": True,
        "interface": get_nebula_interface(),
        "ip": get_nebula_ip(),
        "lighthouses": get_nebula_lighthouse_ips(),
    }
    
    # Try to check if nebula is running
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "nebula"], 
            capture_output=True, 
            text=True
        )
        result["running"] = proc.returncode == 0
    except Exception:
        result["running"] = None
    
    return result

class NetworkRetryHandler:
    """
    Utility class for handling retries in unreliable network environments.
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        backoff_factor: float = 1.5,
    ):
        """
        Initialize with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay by after each attempt
        """
        self.max_retries = int(os.environ.get("TORCHFT_CONNECT_RETRY", str(max_retries)))
        self.retry_delay = float(os.environ.get("TORCHFT_CONNECT_TIMEOUT", str(retry_delay)))
        self.backoff_factor = backoff_factor
        
    def execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            args: Arguments to pass to function
            kwargs: Keyword arguments to pass to function
            
        Returns:
            Result of the function
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        import time
        
        last_error = None
        current_delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor
        
        # If we get here, all retries failed
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

class NebulaCoordinator:
    """
    Helper for coordinating federated learning over Nebula VPN.
    """
    
    def __init__(self, use_lighthouses: bool = True):
        """
        Initialize with Nebula configuration.
        
        Args:
            use_lighthouses: Whether to use lighthouses for coordination
        """
        self.use_lighthouses = use_lighthouses
        self._retry_handler = NetworkRetryHandler()
        
    def get_known_hosts(self) -> Set[str]:
        """
        Get the set of known host IPs on the Nebula network.
        
        Returns:
            Set of IP addresses
        """
        hosts = set()
        
        # Add lighthouse IPs
        if self.use_lighthouses:
            hosts.update(get_nebula_lighthouse_ips())
        
        # Add hosts from hosts file
        hosts.update(get_nebula_hosts().keys())
        
        # Add hosts from environment variable if specified
        env_hosts = os.environ.get("TORCHFT_NEBULA_PEERS", "")
        if env_hosts:
            hosts.update(ip.strip() for ip in env_hosts.split(","))
            
        return hosts
    
    def discover_services(self, port: int = 0) -> Dict[str, str]:
        """
        Discover other federated learning services on the Nebula network.
        
        Args:
            port: Port to check for services (0 to use environment variable)
            
        Returns:
            Dictionary mapping IP addresses to service URLs
        """
        import concurrent.futures
        
        if port == 0:
            port = int(os.environ.get("TORCHFT_SERVICE_PORT", "29500"))
            
        hosts = self.get_known_hosts()
        services = {}
        
        def check_host(host: str) -> Optional[Tuple[str, str]]:
            url = f"http://{host}:{port}/status"
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return (host, f"http://{host}:{port}")
            except Exception:
                pass
            return None
        
        # Check hosts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_host, host) for host in hosts]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    services[result[0]] = result[1]
        
        return services