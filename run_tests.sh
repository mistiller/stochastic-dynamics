#!/bin/bash
set -e

# Function to run tests for a specific package
run_package_tests() {
    local package_name=$1
    local package_path=$2
    
    echo "Building test environment for $package_name..."
    docker build -f Dockerfile.test --build-arg PACKAGE_PATH="$package_path" -t path-integral-optimizer-test .
    
    echo "Running $package_name tests..."
    docker run --rm path-integral-optimizer-test
}

# Run tests for both packages
echo "Testing knapsack-optimizer..."
run_package_tests "knapsack-optimizer" "knapsack-optimizer"

echo -e "\nTesting path-integral-optimizer..."
run_package_tests "path-integral-optimizer" "path-integral-optimizer"

echo -e "\nAll tests passed successfully!"
