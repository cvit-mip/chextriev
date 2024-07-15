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

# anaxnet attn
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --dropout 0.1 \
    --graph_importance 0.2 \
    --is_global_feat \
    --task anaxnet_attn \
    --run anaxnet_attn \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --graph_importance 0.2 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.2_global_feat_lr_1e-4 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.001 \
    --grad_accum 4 \
    --graph_importance 0.2 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.2_global_feat_lr_1e-3 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --graph_importance 0.2 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.2_global_feat_lr_5e-4 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --graph_importance 0.1 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.1_global_feat_lr_1e-4 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --graph_importance 0.5 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.5_global_feat_lr_1e-4 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log

# anaxnet custom: multitask learning, global feature
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --graph_importance 0.8 \
    --is_global_feat \
    --task anaxnet_custom \
    --run anaxnet_gi_0.8_global_feat_lr_1e-4 \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log


# anaxnet custom: multitask learning
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --task anaxnet_custom \
#     --run anaxnet_custom_resnet50_finetuned \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log


# # anaxnet
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --task anaxnet \
#     --run anaxnet \
#     --gpu_ids 0 1 \
#     --num_workers 20 \
#     --train \
#     --log


# anaxnet
python main.py \
    --num_classes 9 \
    --batch_size 32 \
    --lr 0.0001 \
    --grad_accum 4 \
    --task anaxnet \
    --run anaxnet_final \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet_final \
    --num_workers 32


# best config on local features
python main.py \
    --num_classes 9 \
    --batch_size 32 \
    --lr 0.0001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --residual_type 2 \
    --num_nodes 25 \
    --task local_features \
    --run best_config_gt_25_nodes \
    --gpu_ids 0 1 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 32 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --residual_type 2 \
    --num_nodes 25 \
    --task local_features \
    --run best_config_gt_25_nodes \
    --num_workers 32
