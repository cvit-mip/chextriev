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

python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_exp \
    --num_workers 32

# resnet50 final fc finetune
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --task resnet50 \
    --run resnet50_fc \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# densenet121 final fc finetune
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --task densenet121 \
    --run densenet121_fc \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
