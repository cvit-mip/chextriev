#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J train077
#SBATCH --reservation naren_MICCAI
#SBATCH --output=/tmp/train_077.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode077
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1


# best config on local features
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
    --residual_type 2 \
    --num_nodes 6 \
    --task local_features \
    --run best_config_gt \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
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
    --residual_type 2 \
    --num_nodes 6 \
    --task local_features \
    --run best_config_gt \
    --num_workers 32


# # best config with edges and global gated residue, without positional embeddings
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 2 \
#     --run best_config_with_edges_without_pos_emb \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 2 \
#     --run best_config_with_edges_without_pos_emb \
#     --num_workers 32

# # best config with learnt embeddings and edges and local and global gated residue
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 3 \
#     --run best_config_abs_pos_with_edges_local_and_global_gated_residue \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 3 \
#     --run best_config_abs_pos_with_edges_local_and_global_gated_residue \
#     --num_workers 32

# # best config with learnt embeddings and edges and local gated residue
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 1 \
#     --run best_config_abs_pos_with_edges_local_gated_residue \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --residual_type 1 \
#     --run best_config_abs_pos_with_edges_local_gated_residue \
#     --num_workers 32

# # resnet50 final fc finetune
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --grad_accum 4 \
#     --graph_importance 0.0 \
#     --task resnet50 \
#     --run resnet50_fc_node_level \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --graph_importance 0.0 \
#     --task resnet50 \
#     --run resnet50_fc_node_level \
#     --num_workers 32

# # vanilla transformer
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task vanilla_transformer \
#     --run vanilla_transformer_global_lbl_fully_connected \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task vanilla_transformer \
#     --run vanilla_transformer_global_lbl_fully_connected \
#     --num_workers 32

# # best config with learnt embeddings and edges
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --run best_config_abs_pos_with_edges \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --task graph_transformer \
#     --run best_config_abs_pos_with_edges \
#     --num_workers 32

# # best config with learnt embeddings, edges, and global feature
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --is_global_feat \
#     --task graph_transformer \
#     --run best_config_abs_pos_with_edges_global_feat \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --accept_edges \
#     --is_global_feat \
#     --task graph_transformer \
#     --run best_config_abs_pos_with_edges_global_feat \
#     --num_workers 32

# # best config with learnt embeddings
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --task graph_transformer \
#     --run best_config_abs_pos \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --abs_pos \
#     --task graph_transformer \
#     --run best_config_abs_pos \
#     --num_workers 32

# # best config with rotary embeddings
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --rel_pos \
#     --task graph_transformer \
#     --run best_config_rotary \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --rel_pos \
#     --task graph_transformer \
#     --run best_config_rotary \
#     --num_workers 32

# # best config with label co-occurence adjacency matrix
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --task graph_transformer \
#     --run best_config_adj_mat \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --task graph_transformer \
#     --run best_config_adj_mat \
#     --num_workers 32

# # 2 layer fully connected graph transformer with cls token multitask
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --is_global_feat \
#     --cls \
#     --task graph_transformer \
#     --run gt_fc_2_layer_cls_multitask \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --is_global_feat \
#     --cls \
#     --task graph_transformer \
#     --run gt_fc_2_layer_cls_multitask \
#     --num_workers 32

# # 2 layer fully connected graph transformer with cls token
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --is_global_feat \
#     --cls \
#     --task graph_transformer \
#     --run gt_fc_2_layer_cls \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --is_global_feat \
#     --cls \
#     --task graph_transformer \
#     --run gt_fc_2_layer_cls \
#     --num_workers 32

# # 2 layer fully connected graph transformer global image level classification no global features
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_2_layer_no_global_feat_image_only_classification \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_2_layer_no_global_feat_image_only_classification \
#     --num_workers 32

# # 3 layer fully connected graph transformer global image level classification no global features
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 3 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_3_layer_no_global_feat_image_only_classification \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 3 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_3_layer_no_global_feat_image_only_classification \
#     --num_workers 32

# # 4 layer fully connected graph transformer global image level classification no global features
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 4 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_4_layer_no_global_feat_image_only_classification \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 4 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_4_layer_no_global_feat_image_only_classification \
#     --num_workers 32

# # fully connected graph transformer without global features
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_no_global_feat \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_no_global_feat \
#     --num_workers 32

# # fully connected graph transformer cosine lr scheduler
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --lr_scheduler cosine \
#     --task graph_transformer \
#     --run gt_fc \
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
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --lr_scheduler cosine \
#     --task graph_transformer \
#     --run gt_fc \
#     --num_workers 32

# # xfactor mean pool nodes, global image level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --pool mean \
#     --minimalistic \
#     --task xfactor \
#     --run mean_pool_global_image_classification_bz \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --pool mean \
#     --minimalistic \
#     --task xfactor \
#     --run mean_pool_global_image_classification_bz \
#     --num_workers 32

# # graph transformer with rotary embeddings
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --rot_pos \
#     --task graph_transformer \
#     --run gt_rot_pos \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --rot_pos \
#     --task graph_transformer \
#     --run gt_rot_pos \
#     --num_workers 32
