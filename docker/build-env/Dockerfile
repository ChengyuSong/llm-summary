# Build environment for C/C++ projects with LLVM 18 toolchain
# Optimized for static analysis with LTO and LLVM IR generation

FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install LLVM 18 toolchain and build essentials
RUN apt-get update && apt-get install -y \
    # LLVM 18 toolchain
    clang-18 \
    clang++-18 \
    llvm-18 \
    llvm-18-dev \
    llvm-18-tools \
    lld-18 \
    libc++-18-dev \
    libc++abi-18-dev \
    # Build systems
    cmake \
    ninja-build \
    make \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # Version control
    git \
    # Common dependencies
    zlib1g-dev \
    libssl-dev \
    libpng-dev \
    libjpeg-dev \
    # Build speedup
    ccache \
    # Utilities
    wget \
    curl \
    file \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set clang-18 as default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-18 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-18 100 && \
    update-alternatives --install /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-18 100 && \
    update-alternatives --install /usr/bin/opt opt /usr/bin/opt-18 100

# Set up ccache for faster rebuilds
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR="/workspace/.ccache"

# Default compiler environment variables
ENV CC=clang-18
ENV CXX=clang++-18
ENV AR=llvm-ar-18
ENV NM=llvm-nm-18
ENV RANLIB=llvm-ranlib-18
ENV LD=ld.lld-18

# Working directory for builds
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
