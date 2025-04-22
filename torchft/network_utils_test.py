#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test module for network utilities
"""

import os
import unittest
from unittest import mock
import socket
import requests
import time
from typing import Dict, Any

from torchft.network_utils import (
    get_external_ip,
    is_nebula_enabled,
    get_nebula_interface,
    get_nebula_ip,
    check_nebula_status,
    NetworkRetryHandler,
    NebulaCoordinator,
)

class TestNetworkUtils(unittest.TestCase):
    """Test case for network utilities."""
    
    def setUp(self):
        # Save original environment variables
        self.orig_env = dict(os.environ)
    
    def tearDown(self):
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.orig_env)
    
    def test_is_nebula_enabled(self):
        """Test Nebula enabled detection."""
        # Test disabled by default
        self.assertFalse(is_nebula_enabled())
        
        # Test various true values
        for value in ["1", "true", "yes", "TRUE", "Yes"]:
            os.environ["TORCHFT_USE_NEBULA"] = value
            self.assertTrue(is_nebula_enabled())
        
        # Test various false values
        for value in ["0", "false", "no", "FALSE", "No", ""]:
            os.environ["TORCHFT_USE_NEBULA"] = value
            self.assertFalse(is_nebula_enabled())
    
    def test_get_external_ip_from_env(self):
        """Test getting external IP from environment variable."""
        test_ip = "192.168.100.1"
        os.environ["TORCHFT_EXTERNAL_IP"] = test_ip
        self.assertEqual(get_external_ip(), test_ip)
    
    @mock.patch('requests.get')
    def test_get_external_ip_from_service(self, mock_get):
        """Test getting external IP from external service."""
        test_ip = "203.0.113.1"
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = test_ip
        mock_get.return_value = mock_response
        
        self.assertEqual(get_external_ip(), test_ip)
        # Should have tried at least one service
        mock_get.assert_called()
    
    @mock.patch('requests.get', side_effect=requests.exceptions.RequestException("Test error"))
    @mock.patch('socket.gethostname')
    def test_get_external_ip_fallback(self, mock_hostname, mock_get):
        """Test fallback to hostname when services fail."""
        test_hostname = "test-host"
        mock_hostname.return_value = test_hostname
        
        self.assertEqual(get_external_ip(), test_hostname)
        # Should have tried services
        mock_get.assert_called()
        # Should have fallen back to hostname
        mock_hostname.assert_called_once()
    
    def test_get_nebula_interface_from_env(self):
        """Test getting Nebula interface from environment variable."""
        test_interface = "nebula1"
        os.environ["TORCHFT_NEBULA_INTERFACE"] = test_interface
        self.assertEqual(get_nebula_interface(), test_interface)
    
    @mock.patch('subprocess.run')
    def test_get_nebula_interface_autodetect(self, mock_run):
        """Test auto-detection of Nebula interface."""
        # Mock subprocess output containing the interface name
        mock_process = mock.Mock()
        mock_process.returncode = 0
        mock_process.stdout = "1: lo: <LOOPBACK,UP> mtu 65536\n2: nebula1: <POINTOPOINT,UP> mtu 1300"
        mock_run.return_value = mock_process
        
        self.assertEqual(get_nebula_interface(), "nebula1")
        mock_run.assert_called_once()
    
    def test_network_retry_handler(self):
        """Test the NetworkRetryHandler class."""
        handler = NetworkRetryHandler(max_retries=3, retry_delay=0.1)
        
        # Test successful execution
        result = handler.execute_with_retry(lambda: "success")
        self.assertEqual(result, "success")
        
        # Test retrying on failure
        fail_count = [0]
        def fail_twice_then_succeed():
            fail_count[0] += 1
            if fail_count[0] <= 2:
                raise ValueError("Temporary error")
            return "success after retry"
        
        result = handler.execute_with_retry(fail_twice_then_succeed)
        self.assertEqual(result, "success after retry")
        self.assertEqual(fail_count[0], 3)
        
        # Test failure after max retries
        with self.assertRaises(RuntimeError):
            handler.execute_with_retry(lambda: 1/0)
    
    def test_network_retry_handler_env_override(self):
        """Test environment variable override for NetworkRetryHandler."""
        os.environ["TORCHFT_CONNECT_RETRY"] = "5"
        os.environ["TORCHFT_CONNECT_TIMEOUT"] = "2.5"
        
        handler = NetworkRetryHandler()
        self.assertEqual(handler.max_retries, 5)
        self.assertEqual(handler.retry_delay, 2.5)
    
    @mock.patch('torchft.network_utils.get_nebula_interface')
    @mock.patch('torchft.network_utils.get_nebula_ip')
    @mock.patch('torchft.network_utils.get_nebula_lighthouse_ips')
    def test_check_nebula_status(self, mock_lighthouse, mock_ip, mock_interface):
        """Test checking Nebula status."""
        os.environ["TORCHFT_USE_NEBULA"] = "1"
        mock_interface.return_value = "nebula1"
        mock_ip.return_value = "192.168.100.1"
        mock_lighthouse.return_value = ["192.168.100.254"]
        
        status = check_nebula_status()
        
        self.assertTrue(status["enabled"])
        self.assertEqual(status["interface"], "nebula1")
        self.assertEqual(status["ip"], "192.168.100.1")
        self.assertEqual(status["lighthouses"], ["192.168.100.254"])
    
    @mock.patch('torchft.network_utils.NebulaCoordinator.get_known_hosts')
    @mock.patch('concurrent.futures.ThreadPoolExecutor')
    def test_nebula_coordinator_discover(self, mock_executor, mock_get_hosts):
        """Test NebulaCoordinator service discovery."""
        mock_get_hosts.return_value = {"192.168.100.1", "192.168.100.2"}
        
        # Set up mock executor
        mock_future1 = mock.Mock()
        mock_future1.result.return_value = ("192.168.100.1", "http://192.168.100.1:29501")
        
        mock_future2 = mock.Mock()
        mock_future2.result.return_value = None
        
        mock_executor_instance = mock.Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]
        mock_executor.return_value = mock_executor_instance
        
        # Create coordinator and discover services
        coordinator = NebulaCoordinator()
        services = coordinator.discover_services()
        
        # Verify results
        self.assertEqual(services, {"192.168.100.1": "http://192.168.100.1:29501"})
        mock_get_hosts.assert_called_once()

if __name__ == '__main__':
    unittest.main()