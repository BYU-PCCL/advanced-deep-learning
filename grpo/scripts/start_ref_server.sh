source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Accept port as argument, default to 59875
PORT=${1:-59875}

uv run ref_server.py --port $PORT