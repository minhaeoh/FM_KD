#!/bin/sh
#SBATCH -J model
#SBATCH -N 1            
#SBATCH -p gpu02       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=12
#SBATCH -o model/%j.out 



cd /home/minhae/diffusion/FM_KD

python script/test_student_model_new.py \
  --device cuda \
  --teacher-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --student-num-layers 16 \
  --anchor-layers 0,7,15 \
  --include-final \
  --run-text-demo \
  --prompt "What is the capital of France?" \
  --gen-len 32 \
  --ode-steps 32 \
  --temperature 0.0