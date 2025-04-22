#!/bin/bash
set -e

# Start Nebula in the background
nebula -config /app/config.yml &

# Wait for Nebula to start
sleep 2

# Print Nebula status
ip addr show nebula1

echo "Nebula Started Successfully!"
echo "You can now execute commands in this container."

# Keep container running
tail -f /dev/null