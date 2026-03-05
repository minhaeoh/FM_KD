#!/bin/bash
#SBATCH -J train_dbg_fsdp
#SBATCH -N 1
#SBATCH -p gpu04
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o train/%j.out


cd /home/minhae/diffusion/FM_KD

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train_fsdp.py \
  --student-init-ckpt /home/minhae/diffusion/FM_KD/checkpoints/student_init.pt \
  --data-root /home/minhae/diffusion/FM_KD/data/collected_data \
  --lambda-kl 0.0 \
  --batch-size 1 \
  --num-workers 2 \
  --bf16 \
  --activation-checkpointing \
  --output-dir /home/minhae/diffusion/FM_KD/checkpoints_cfd_fsdp \
  --run-name cfd_fsdp_2gpu