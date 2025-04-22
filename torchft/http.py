import logging
import os
import socket
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple, Union, Dict, Any, Type
from functools import partial

logger: logging.Logger = logging.getLogger(__name__)

class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family: socket.AddressFamily = socket.AF_INET6
    request_queue_size: int = 1024
    
    def server_bind(self) -> None:
        """Override server_bind to ensure we can rebind to the same address."""
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()
        
class HTTPServerFactory:
    """Factory for creating HTTP servers that work in various network environments."""
    
    @staticmethod
    def create_server(
        server_address: Tuple[str, int],
        handler_class,
        nebula_mode: bool = False,
    ) -> Union[ThreadingHTTPServer, _IPv6HTTPServer]:
        """
        Create an HTTP server suitable for the current network environment.
        
        Args:
            server_address: (host, port) tuple
            handler_class: Request handler class
            nebula_mode: Whether to use configuration suitable for Nebula networks
            
        Returns:
            Configured HTTP server instance
        """
        from torchft.network_utils import is_nebula_enabled
        
        # Check if explicitly enabled via env var
        if nebula_mode or is_nebula_enabled():
            logger.info("Using Nebula-optimized HTTP server configuration")
            # For Nebula, use IPv4 by default and allow larger connection queue
            server = ThreadingHTTPServer(server_address, handler_class)
            server.request_queue_size = 2048  # Larger queue for unreliable connections
            
            # Apply socket options for unreliable networks
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set longer timeouts for unreliable networks
            timeout = int(os.environ.get("TORCHFT_SOCKET_TIMEOUT", "120"))
            server.socket.settimeout(timeout)
            
            return server
        else:
            # Use IPv6 server by default for datacenter environments
            server = _IPv6HTTPServer(server_address, handler_class)
            return server

class StatusHandler(BaseHTTPRequestHandler):
    """Handler for serving federated learning node status."""
    
    # Will be set with additional status info
    status_info: Dict[str, Any] = {}
    
    def do_GET(self):
        """Handle GET request to status endpoint."""
        if self.path == "/status":
            self._handle_status()
        else:
            self.send_error(404, "Not found")
    
    def _handle_status(self):
        """Return status information for this node."""
        from torchft.network_utils import check_nebula_status, get_external_ip
        
        # Build basic status information
        status = {
            "status": "ok",
            "network": {
                "hostname": socket.gethostname(),
                "external_ip": get_external_ip(),
            },
            "nebula": check_nebula_status(),
        }
        
        # Add any additional status information
        status.update(self.status_info)
        
        # Send the response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode("utf-8"))
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.debug(f"{self.address_string()} - {format%args}")

class StatusServer:
    """
    A simple HTTP server for node discovery and status in federated learning.
    
    This server runs in the background and provides basic status information
    about this node for other nodes to discover.
    """
    
    def __init__(
        self, 
        port: int = 0,
        status_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the status server.
        
        Args:
            port: Port to listen on (0 for auto-assign)
            status_info: Additional status information to include
        """
        self.port = port or int(os.environ.get("TORCHFT_STATUS_PORT", "29501"))
        
        # Create a custom handler with our status info
        handler_class = type('CustomStatusHandler', (StatusHandler,), {
            'status_info': status_info or {}
        })
        
        # Create the server
        from torchft.network_utils import is_nebula_enabled
        self.server = HTTPServerFactory.create_server(
            ("", self.port),
            handler_class,
            nebula_mode=is_nebula_enabled()
        )
        
        # Update port if auto-assigned
        if port == 0:
            self.port = self.server.socket.getsockname()[1]
        
        # Start server in a background thread
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        )
    
    def start(self):
        """Start the status server."""
        self.thread.start()
        logger.info(f"Started status server on port {self.port}")
        
    def stop(self):
        """Stop the status server."""
        self.server.shutdown()
        self.thread.join()
        logger.info("Stopped status server")
    
    def update_status(self, status_info: Dict[str, Any]):
        """
        Update the status information.
        
        Args:
            status_info: New status information to include
        """
        self.server.RequestHandlerClass.status_info.update(status_info)
        
    def get_url(self) -> str:
        """
        Get the URL for this status server.
        
        Returns:
            URL for this status server
        """
        from torchft.network_utils import get_external_ip
        return f"http://{get_external_ip()}:{self.port}"
