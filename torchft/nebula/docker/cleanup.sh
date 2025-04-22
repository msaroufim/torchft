#!/bin/bash
set -e

echo "Stopping and removing containers..."
docker stop lighthouse worker || true
docker rm lighthouse worker || true

echo "Removing Docker network..."
docker network rm nebula-test || true

echo "Cleanup complete!"
echo "If you want to completely remove the test environment, you can also delete the ~/nebula-test directory."