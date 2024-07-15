#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J train_hyp_077
#SBATCH --reservation naren_MICCAI
#SBATCH --output=/tmp/train_hyp_077.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode077
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.995, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.995 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.995_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.995 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.995_global_mean \
    --num_workers 32

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.98, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.98 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.98_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.98 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.98_global_mean \
    --num_workers 32

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.95, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.95 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.95_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.95 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.95_global_mean \
    --num_workers 32

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.8, GLOBAL_FEAT, MEAN_POOL
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.8 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.8_global_mean \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.8 \
    --fully_connected \
    --is_global_feat \
    --pool mean \
    --task graph_transformer \
    --run hp_l2_gi0.8_global_mean \
    --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.995
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.995 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.995 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.995 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.995 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.98
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.98 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.98 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.98 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.98 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.95
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.95 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.95 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.95 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.95 \
#     --num_workers 32

# # # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.50
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.50 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.50 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.50 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.50 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.05
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.05 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.05 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.05 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.05 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 0.2
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.2 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_l2_gi0.2 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 16 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_256 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_256 \
#     --num_workers 32

# # BZ: 16*16, LR: 0.0001, DROPOUT: 0.1, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 16 \
#     --dropout 0.1 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_256_drop_0.1 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_256_drop_0.1 \
#     --num_workers 32

# # BZ: 16*32, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_512 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_512 \
#     --num_workers 32

# # BZ: 16*32, LR: 0.0001, DROPOUT: 0.1, NUM_LAYERS: 1, GRAPH_IMPORTANCE: 1.0
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.1 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_512_drop_0.1 \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run hp_bz_512_drop_0.1 \
#     --num_workers 32
