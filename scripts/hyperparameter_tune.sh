#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J train_hyp_046
#SBATCH --reservation arihanth_MIDL
#SBATCH --output=/tmp/train_hyp_046.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode046
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.5, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.5 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.5_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.5 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.5_global_mean \
    --num_workers 32

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.05, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.05 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.05_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.05 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.05_global_mean \
    --num_workers 32

# # BZ: 16*8, LR: 0.006, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.006 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_6e-3 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.006 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_6e-3 \
#     --num_workers 32

# # BZ: 16*8, LR: 0.001, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_1e-3 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_1e-3 \
#     --num_workers 32

# # BZ: 16*8, LR: 0.0004, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0004 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_4e-4 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0004 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_4e-4 \
#     --num_workers 32

# # BZ: 16*8, LR: 0.00008, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.00008 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_8e-5 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.00008 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_lr_8e-5 \
#     --num_workers 32
