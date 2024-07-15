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

export PYTHONUNBUFFERED=1

# gcn attn
python main.py \
    --emb_dim 1024 \
    --edge_dim 32 \
    --num_layers 2 \
    --num_classes 14 \
    --batch_size 16 \
    --lr 0.0006 \
    --grad_accum 1 \
    --task gcn_global \
    --run gcn_global \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 16 \
    --train \
    --log