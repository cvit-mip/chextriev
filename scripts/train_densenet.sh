#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J train_densenet
#SBATCH --reservation arihanth_MIDL
#SBATCH --output=logs/train_densenet.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode046
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

python main.py \
    --num_classes 14 \
    --batch_size 64 \
    --lr 0.0005 \
    --grad_accum 1 \
    --task mimic-cxr-jpg \
    --run densenet_all_images \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
