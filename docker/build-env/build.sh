#!/bin/bash
# Build the llm-summary-builder Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${1:-llm-summary-builder:latest}"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo ""
echo "Build complete!"
echo "Image: $IMAGE_NAME"
echo ""
echo "Test the image with:"
echo "  docker run --rm $IMAGE_NAME clang-18 --version"
echo "  docker run --rm $IMAGE_NAME cmake --version"
