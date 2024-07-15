#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J eval077
#SBATCH --reservation naren_MICCAI
#SBATCH --output=/tmp/eval077.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode077
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

# best config with learnt embeddings and edges: graph transformer
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --task graph_transformer \
    --run best_config_abs_pos_with_edges \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log

python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --task graph_transformer \
    --run best_config_abs_pos_with_edges \
    --num_workers 32

# anaxnet
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 32

# ath
python temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36

# resnet50
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc_new \
    --num_workers 32

# vanilla transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --task vanilla_transformer \
    --run vanilla_transformer_global_lbl_fully_connected \
    --num_workers 32


# best model without positional embedding: graph transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --task graph_transformer \
    --run gt_fc_2_layer_no_global_feat_image_only_classification \
    --num_workers 32

# multitask graph transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.2 \
    --fully_connected \
    --is_global_feat \
    --cls \
    --task graph_transformer \
    --run gt_fc_2_layer_cls_multitask \
    --num_workers 32

# best model with learnt pos emb: graph transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --abs_pos \
    --task graph_transformer \
    --run best_config_abs_pos \
    --num_workers 32

# best model with rotary embeddings: graph transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --rel_pos \
    --task graph_transformer \
    --run best_config_rotary \
    --num_workers 32
