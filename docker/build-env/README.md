# LLVM Build Environment

Docker image for building C/C++ projects with LLVM 18 toolchain, optimized for static analysis.

## Features

- **LLVM 18 Toolchain**: Clang 18, LLVM tools, LLD linker
- **Build Systems**: CMake 3.28+, Ninja, Autotools, Make
- **LTO Support**: Full Link-Time Optimization with `-flto=full`
- **IR Generation**: LLVM bitcode generation with `-emit-llvm` and `-save-temps=obj`
- **Static Analysis Ready**: Generates `compile_commands.json` for libclang
- **Build Caching**: ccache for faster incremental builds

## Building the Image

```bash
./build.sh
# Or with custom name:
./build.sh my-custom-name:tag
```

## Usage

### Interactive Shell

```bash
docker run --rm -it \
  -v /path/to/project:/workspace \
  llm-summary-builder:latest
```

### CMake Build

```bash
docker run --rm \
  -v /path/to/project:/workspace \
  -w /workspace/build \
  llm-summary-builder:latest \
  bash -c "cmake -G Ninja \
           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
           -DCMAKE_C_COMPILER=clang-18 \
           -DCMAKE_CXX_COMPILER=clang++-18 \
           -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
           -DCMAKE_C_FLAGS='-flto=full -save-temps=obj' \
           -DCMAKE_CXX_FLAGS='-flto=full -save-temps=obj' \
           .. && \
           ninja -j\$(nproc)"
```

## Environment Variables

- `CC=clang-18`: Default C compiler
- `CXX=clang++-18`: Default C++ compiler
- `AR=llvm-ar-18`: LLVM archiver for LTO
- `NM=llvm-nm-18`: LLVM symbol table reader
- `RANLIB=llvm-ranlib-18`: LLVM ranlib

## Installed Tools

- `clang-18`, `clang++-18`: C/C++ compilers
- `llvm-link-18`: LLVM bitcode linker
- `llvm-ar-18`: LLVM archiver
- `opt-18`: LLVM optimizer
- `cmake`: Build system generator
- `ninja`: Fast build tool
- `ccache`: Compiler cache
- `git`: Version control

## LTO and IR Generation

This image is configured to support:

1. **Link-Time Optimization (LTO)**:
   - `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`
   - `-flto=full` for full LTO mode

2. **LLVM IR Generation**:
   - `-save-temps=obj` saves intermediate files including `.bc` (bitcode) and `.ll` (LLVM IR text)
   - IR files can be used for advanced static and dynamic analysis

3. **Static Linking Preference**:
   - `-DBUILD_SHARED_LIBS=OFF` for CMake projects
   - Produces standalone binaries with all dependencies linked in

## Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
docker run --rm \
  -v /path/to/project:/workspace \
  -u $(id -u):$(id -g) \
  llm-summary-builder:latest
```

### ccache Not Working

Ensure ccache directory is writable:

```bash
docker run --rm \
  -v /path/to/project:/workspace \
  -v /path/to/.ccache:/workspace/.ccache \
  llm-summary-builder:latest
```
