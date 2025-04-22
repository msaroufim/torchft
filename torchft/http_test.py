#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test module for HTTP server utilities.
"""

import os
import unittest
from unittest import mock
import socket
import json
import threading
import requests
import time
from http.server import BaseHTTPRequestHandler

from torchft.http import (
    HTTPServerFactory,
    StatusHandler,
    StatusServer,
)

class TestHTTPServer(unittest.TestCase):
    """Test case for HTTP server utilities."""
    
    def setUp(self):
        # Save original environment variables
        self.orig_env = dict(os.environ)
    
    def tearDown(self):
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.orig_env)
    
    @mock.patch('torchft.http.ThreadingHTTPServer')
    @mock.patch('torchft.http._IPv6HTTPServer')
    @mock.patch('torchft.network_utils.is_nebula_enabled')
    def test_http_server_factory_nebula(self, mock_is_nebula, mock_ipv6, mock_threading):
        """Test HTTP server factory with Nebula enabled."""
        mock_is_nebula.return_value = True
        
        # Create a dummy handler class
        handler_class = type('DummyHandler', (BaseHTTPRequestHandler,), {})
        
        # Call factory method
        HTTPServerFactory.create_server(("", 8000), handler_class)
        
        # Should have created threading HTTP server (for Nebula), not IPv6
        mock_threading.assert_called_once()
        mock_ipv6.assert_not_called()
        
        # Verify socket options for residential networks
        server_instance = mock_threading.return_value
        server_instance.socket.setsockopt.assert_called_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        
        # Should have set timeout
        server_instance.socket.settimeout.assert_called_once()
    
    @mock.patch('torchft.http.ThreadingHTTPServer')
    @mock.patch('torchft.http._IPv6HTTPServer')
    @mock.patch('torchft.network_utils.is_nebula_enabled')
    def test_http_server_factory_datacenter(self, mock_is_nebula, mock_ipv6, mock_threading):
        """Test HTTP server factory in datacenter mode."""
        mock_is_nebula.return_value = False
        
        # Create a dummy handler class
        handler_class = type('DummyHandler', (BaseHTTPRequestHandler,), {})
        
        # Call factory method
        HTTPServerFactory.create_server(("", 8000), handler_class)
        
        # Should have created IPv6 HTTP server, not threading
        mock_ipv6.assert_called_once()
        mock_threading.assert_not_called()
    
    @mock.patch('torchft.http.HTTPServerFactory.create_server')
    @mock.patch('torchft.network_utils.get_external_ip')
    def test_status_server(self, mock_get_ip, mock_create_server):
        """Test StatusServer class."""
        mock_get_ip.return_value = "192.168.1.100"
        mock_server = mock.Mock()
        mock_create_server.return_value = mock_server
        
        # Create status server
        test_port = 8080
        test_status = {"role": "coordinator", "status": "ready"}
        server = StatusServer(port=test_port, status_info=test_status)
        
        # Check that server was created correctly
        mock_create_server.assert_called_once()
        args, kwargs = mock_create_server.call_args
        self.assertEqual(args[0], ("", test_port))
        self.assertTrue(issubclass(args[1], StatusHandler))
        
        # Check initial status
        self.assertEqual(server.server.RequestHandlerClass.status_info, test_status)
        
        # Test starting server
        server.start()
        self.assertTrue(server.thread.is_alive())
        
        # Test stopping server
        server.stop()
        mock_server.shutdown.assert_called_once()
        
        # Test URL generation
        self.assertEqual(server.get_url(), f"http://192.168.1.100:{test_port}")
        
        # Test status update
        new_status = {"connected_workers": 5}
        server.update_status(new_status)
        expected_status = test_status.copy()
        expected_status.update(new_status)
        self.assertEqual(
            server.server.RequestHandlerClass.status_info, 
            expected_status
        )

if __name__ == '__main__':
    unittest.main()