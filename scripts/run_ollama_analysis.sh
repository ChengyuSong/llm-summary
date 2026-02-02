#!/bin/bash
# Script to run llm-summary commands with Ollama in Docker

set -e  # Exit on error

# Configuration
OLLAMA_CONTAINER="ollama"
OLLAMA_PORT="11434"
OLLAMA_MODEL="qwen3-coder:30b"
OLLAMA_VOLUME="/docker/ollama-models"
KEEP_ALIVE="24h"
USE_SUDO=""
VENV_PATH="venv"
FORCE_RESTART=false
SKIP_OLLAMA=false
STOP_OLLAMA=false

# Parse command line arguments
COMMAND=""
LLM_ARGS=""
SKIP_PULL=false

# First argument is the command (analyze, extract, show, lookup, etc.)
if [[ $# -eq 0 ]]; then
    COMMAND=""
else
    # Check if first arg is a flag or a command
    if [[ "$1" != --* ]]; then
        COMMAND="$1"
        shift
    fi
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --volume-path)
            OLLAMA_VOLUME="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --skip-pull)
            SKIP_PULL=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --stop-ollama)
            STOP_OLLAMA=true
            shift
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        --keep-alive)
            KEEP_ALIVE="$2"
            shift 2
            ;;
        *)
            LLM_ARGS="$LLM_ARGS $1"
            shift
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  analyze              Analyze C/C++ source files"
    echo "  extract              Extract functions and build call graph"
    echo "  show                 Display stored summaries"
    echo "  lookup               Look up a specific function"
    echo "  stats                Show database statistics"
    echo "  callgraph            Export call graph"
    echo "  export               Export all summaries to JSON"
    echo ""
    echo "Ollama Options:"
    echo "  --model MODEL        Ollama model to use (default: qwen3-coder:30b)"
    echo "  --volume-path PATH   Volume path for models (default: /docker/ollama-models)"
    echo "  --skip-pull          Skip pulling Docker image and model"
    echo "  --skip-ollama        Skip Ollama setup entirely (for commands that don't need LLM)"
    echo "  --force-restart      Force restart container even if running"
    echo "  --stop-ollama        Stop and remove Ollama container after command completes"
    echo "  --keep-alive TIME    Keep model loaded (default: 24h)"
    echo ""
    echo "Script Options:"
    echo "  --venv PATH          Path to Python venv (default: venv)"
    echo ""
    echo "Command-specific options (passed to llm-summary):"
    echo "  --path PATH          Path to analyze (for analyze/extract)"
    echo "  --db PATH            Database file path"
    echo "  --backend TYPE       LLM backend (claude, openai, ollama)"
    echo "  --verbose            Verbose output"
    echo "  --force              Force re-analysis"
    echo "  ... and more         See llm-summary <command> --help"
    echo ""
    echo "Examples:"
    echo "  $0 analyze --path /path/to/project --verbose"
    echo "  $0 show --allocating-only --db project.db"
    echo "  $0 lookup create_buffer --db project.db"
    echo "  $0 stats --db project.db --skip-ollama"
    exit 1
fi

echo "=== llm-summary: $COMMAND ==="

# Determine if we need Ollama based on command and flags
NEEDS_OLLAMA=true
case "$COMMAND" in
    show|lookup|stats|export|callgraph)
        # These commands don't need LLM
        NEEDS_OLLAMA=false
        ;;
    analyze|extract)
        # These might need LLM depending on backend
        # Default to needing Ollama unless --skip-ollama is set
        ;;
esac

if [ "$SKIP_OLLAMA" = true ]; then
    NEEDS_OLLAMA=false
fi

if [ "$NEEDS_OLLAMA" = true ]; then
    echo "Model: $OLLAMA_MODEL"
    echo "Volume path: $OLLAMA_VOLUME"
    echo "Keep alive: $KEEP_ALIVE"
    echo ""

    # Detect if we need sudo for docker
    if docker ps > /dev/null 2>&1; then
        USE_SUDO=""
    else
        echo "Docker requires sudo, will use sudo for docker commands"
        USE_SUDO="sudo"
    fi

    # Ensure volume directory exists (requires sudo)
    if [ ! -d "$OLLAMA_VOLUME" ]; then
        echo "Creating volume directory: $OLLAMA_VOLUME"
        sudo mkdir -p "$OLLAMA_VOLUME"
    fi

    # Step 1: Pull latest Ollama Docker image
    if [ "$SKIP_PULL" = false ]; then
        echo "[1/6] Pulling latest Ollama Docker image..."
        ${USE_SUDO} docker pull ollama/ollama
    else
        echo "[1/6] Skipping Docker image pull"
    fi
else
    echo "Skipping Ollama setup"
    echo ""
fi

if [ "$NEEDS_OLLAMA" = true ]; then
    # Step 2: Start or restart Ollama container
    echo "[2/6] Checking Ollama container status..."

    CONTAINER_RUNNING=false
    CONTAINER_EXISTS=false

    # Check if container exists and is running
    if ${USE_SUDO} docker ps --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
        CONTAINER_RUNNING=true
        CONTAINER_EXISTS=true
    elif ${USE_SUDO} docker ps -a --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
        CONTAINER_EXISTS=true
    fi

    SHOULD_RESTART=false

    if [ "$CONTAINER_RUNNING" = true ]; then
        if [ "$FORCE_RESTART" = true ]; then
            echo "Container '$OLLAMA_CONTAINER' is running. Force restart requested."
            SHOULD_RESTART=true
            echo "Stopping and removing existing container..."
            ${USE_SUDO} docker rm -f "$OLLAMA_CONTAINER" 2>/dev/null || true
            CONTAINER_EXISTS=false
        else
            echo "Container '$OLLAMA_CONTAINER' is already running. Using existing container."
        fi
    elif [ "$CONTAINER_EXISTS" = true ]; then
        echo "Container '$OLLAMA_CONTAINER' exists but is not running. Removing..."
        ${USE_SUDO} docker rm -f "$OLLAMA_CONTAINER" 2>/dev/null || true
        CONTAINER_EXISTS=false
    fi

    # Start new container if needed
    if [ "$CONTAINER_EXISTS" = false ] || [ "$SHOULD_RESTART" = true ]; then
        echo "Starting new Ollama container..."
        ${USE_SUDO} docker run -d \
            --name "$OLLAMA_CONTAINER" \
            -p "${OLLAMA_PORT}:11434" \
            -v "${OLLAMA_VOLUME}:/root/.ollama" \
            ollama/ollama
    fi

    # Wait for Ollama to be ready (only if we started/restarted container)
    if [ "$CONTAINER_EXISTS" = false ] || [ "$SHOULD_RESTART" = true ]; then
        echo "[3/6] Waiting for Ollama to be ready..."
        MAX_RETRIES=30
        RETRY_COUNT=0
        while ! curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null 2>&1; do
            sleep 1
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
                echo "Error: Ollama failed to start after ${MAX_RETRIES} seconds"
                ${USE_SUDO} docker logs "$OLLAMA_CONTAINER"
                exit 1
            fi
        done
        echo "Ollama is ready!"
    else
        echo "[3/6] Container already running, skipping wait."
    fi

    # Step 3: Pull the model
    if [ "$SKIP_PULL" = false ]; then
        echo "[4/6] Checking if model exists..."

        # Check if model is already pulled
        if ${USE_SUDO} docker exec "$OLLAMA_CONTAINER" ollama list | grep -q "$OLLAMA_MODEL"; then
            echo "Model $OLLAMA_MODEL already exists, skipping pull."
        else
            echo "Pulling model: $OLLAMA_MODEL"
            echo "This may take a while for large models..."
            ${USE_SUDO} docker exec "$OLLAMA_CONTAINER" ollama pull "$OLLAMA_MODEL"
        fi
    else
        echo "[4/6] Skipping model pull (--skip-pull specified)"
    fi

    # Step 4: Pre-load the model into memory
    echo "[5/6] Pre-loading model into memory..."
    curl -s http://localhost:${OLLAMA_PORT}/api/generate -d "{
      \"model\": \"${OLLAMA_MODEL}\",
      \"prompt\": \"\",
      \"keep_alive\": \"${KEEP_ALIVE}\"
    }" > /dev/null

    echo "Model loaded and will stay in memory for $KEEP_ALIVE"
fi

# Activate venv and run the command
echo ""
echo "Running llm-summary command..."

# Find and activate venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/${VENV_PATH}/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Warning: venv not found at $VENV_ACTIVATE"
    echo "Trying to run llm-summary from system PATH..."
else
    echo "Activating venv: $VENV_ACTIVATE"
    source "$VENV_ACTIVATE"
fi

# Build the command
if [ "$NEEDS_OLLAMA" = true ]; then
    # Add Ollama backend flags
    FULL_COMMAND="llm-summary $COMMAND --backend ollama --model \"$OLLAMA_MODEL\" $LLM_ARGS"
else
    FULL_COMMAND="llm-summary $COMMAND $LLM_ARGS"
fi

echo "Command: $FULL_COMMAND"
echo ""

eval $FULL_COMMAND

echo ""
echo "=== Command Complete ==="

if [ "$NEEDS_OLLAMA" = true ]; then
    if [ "$STOP_OLLAMA" = true ]; then
        echo ""
        echo "Stopping and removing Ollama container..."
        ${USE_SUDO} docker stop "$OLLAMA_CONTAINER" > /dev/null
        ${USE_SUDO} docker rm "$OLLAMA_CONTAINER" > /dev/null
        echo "Ollama container stopped and removed."
    else
        echo ""
        echo "Ollama container is still running (use --stop-ollama to stop it)."
        echo "To stop manually: docker stop $OLLAMA_CONTAINER"
        echo "To remove manually: docker rm $OLLAMA_CONTAINER"
    fi
fi
