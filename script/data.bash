#!/bin/sh
#SBATCH -J data
#SBATCH -N 1            
#SBATCH -p gpu01       
#SBATCH --gres=gpu:2
#SBATCH --ntasks=24
#SBATCH -o dataset/%j.out  # %j는 job ID를 의미합니다

python /home/minhae/diffusion/FM_KD/dataset/collect_data.py