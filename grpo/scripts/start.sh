#!/bin/bash
#SBATCH --job-name=grpo_ref_server
#SBATCH --output=logs/grpo_ref_server_%j.out
#SBATCH --error=logs/grpo_ref_server_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --qos=dw87

# Set the reference server port (default: 59875)
export REF_SERVER_PORT=${REF_SERVER_PORT:-59875}

module load cuda/12.4.1-pw6cogp

source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="offline"
mkdir -p /tmp/triton_cache
export TRITON_CACHE_DIR="/tmp/triton_cache"

# Start the reference server in the background, passing the port
bash scripts/start_ref_server.sh $REF_SERVER_PORT > logs/ref_server.log 2>&1 &

# Wait for the server to start
sleep 60

# Start the GRPO process (using 3 GPUs), passing the port
deepspeed --include localhost:2 grpo_lab.py --ref-server-port $REF_SERVER_PORT > logs/grpo.log 2>&1
