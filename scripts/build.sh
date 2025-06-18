#!/usr/bin/env bash

# Load necessary modules based on hostname
module purge
if [[ $(hostname) == *viper* ]]; then
    echo "Loading modules for Viper..."
    module load rocm/6.4
elif [[ $(hostname) == *raven* ]]; then
    echo "Loading modules for Raven..."
    module load gcc/13
    module load cuda/12.6
else
    echo "Error: Unknown hostname. Unable to determine which module to load."
    exit 1
fi

# Default behavior: perform configuration and use all available cores for parallelism
DO_CONFIGURE=true
PARALLEL_CORES=$(nproc) # Default to the number of available cores
BUILD_TYPE="Release"   # Default build type

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-configure)
            DO_CONFIGURE=false
            shift
            ;;
        --parallel)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                PARALLEL_CORES=$2
                shift 2
            else
                echo "Error: --parallel requires a numeric argument."
                exit 1
            fi
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Perform configuration if enabled
if $DO_CONFIGURE; then
    echo "Configuring the project with CMake using $PARALLEL_CORES cores in $BUILD_TYPE mode..."
    rm -rf build
    mkdir build
    cmake -B build -S . -DCMAKE_BUILD_TYPE=$BUILD_TYPE
fi

# Always build the project
echo "Building the project using $PARALLEL_CORES cores..."
cmake --build build --parallel $PARALLEL_CORES
