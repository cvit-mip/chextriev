#!/bin/bash

# resnet50 end to end finetune
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50 \
    --num_workers 20

# anaxnet original checkpoint from dec 26th 2023
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --task anaxnet \
    --run anaxnet \
    --gpu_ids 0 1 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 20

# end to end best config with learnt embeddings and edges
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
    --run end_to_end_best_gt \
    --gpu_ids 0 1 \
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
    --task graph_transformer \
    --run end_to_end_best_gt \
    --num_workers 20

# AnaXNet global (image level classification)
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --pool mean \
    --task graph_benchmark \
    --run anaxnet_global_features_mean_pool \
    --gpu_ids 0 1 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --pool mean \
    --task graph_benchmark \
    --run anaxnet_global_features_mean_pool \
    --num_workers 20

# AnaXNet original (node level classification)
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.0 \
    --pool mean \
    --task graph_benchmark \
    --run anaxnet_local_features_mean_pool \
    --gpu_ids 0 1 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.0 \
    --pool mean \
    --task graph_benchmark \
    --run anaxnet_local_features_mean_pool \
    --num_workers 20

# xfactor mean pool nodes, node level classification
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 1 \
    --graph_importance 0.0 \
    --pool mean \
    --minimalistic \
    --task xfactor \
    --run mean_pool_node_classification_bz \
    --gpu_ids 0 1 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 1 \
    --graph_importance 0.0 \
    --pool mean \
    --minimalistic \
    --task xfactor \
    --run mean_pool_node_classification_bz \
    --num_workers 20

# xfactor mean pool nodes, global image level classification
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 1 \
    --graph_importance 1.0 \
    --pool mean \
    --minimalistic \
    --task xfactor \
    --run mean_pool_global_image_classification_bz \
    --gpu_ids 0 1 2 3 \
    --num_workers 10 \
    --train \
    --log
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 1 \
    --graph_importance 1.0 \
    --pool mean \
    --minimalistic \
    --task xfactor \
    --run mean_pool_global_image_classification_bz \
    --num_workers 32