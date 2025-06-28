#!/bin/bash
set -e

# Build and run the test Docker image
echo "Building test environment..."
docker build -f Dockerfile.test -t path-integral-optimizer-test .

echo "Running tests..."
docker run --rm path-integral-optimizer-test

# Check for test output and report results
if [ $? -eq 0 ]; then
    echo -e "\nAll tests passed successfully!"
else
    echo -e "\nSome tests failed. Please check the output above for details."
    exit 1
fi
