#!/bin/bash
#SBATCH --account=MST111411
#SBATCH --job-name=VideoGrain
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roger@roger.tw

mkdir -p logs

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate videograin

# 設置環境變數處理 cuDNN 兼容性問題
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDNN_USE_FALLBACK=1
export ORT_DISABLE_ALL_OPTIMIZATION=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Job $SLURM_JOB_ID running on $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun python -u /work/rogerfan48/VideoGrain/test.py --config /work/rogerfan48/VideoGrain/config/02_walk_walk_a_config.yaml

# accelerate launch \
#   --mixed_precision='fp16' \
#   --main_process_port 0 \
#   test.py --config config/02_walk_walk_a_config.yaml
