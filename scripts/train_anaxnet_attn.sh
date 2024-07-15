#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J train046
#SBATCH --reservation arihanth_MIDL
#SBATCH --output=/tmp/train_046.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode046
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

# BZ: 16*16, LR: 0.0001, DROPOUT: 0.0, NUM_LAYERS: 2, GRAPH_IMPORTANCE: 1.0
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --multiscale \
    --task graph_transformer \
    --run gt_fc_2_layer_no_global_feat_image_only_classification_multiscale \
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
    --multiscale \
    --task graph_transformer \
    --run gt_fc_2_layer_no_global_feat_image_only_classification_multiscale \
    --num_workers 32

# # efficient net end to end training with 2 layer graph transformer no global features image only classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 4 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --image_featuriser efficientnet \
#     --task graph_transformer \
#     --run efficientnet_end_to_end_2L_no_global_feat_image_lvl \
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
#     --image_featuriser efficientnet \
#     --task graph_transformer \
#     --run efficientnet_end_to_end_2L_no_global_feat_image_lvl \
#     --num_workers 32

# # efficient net end to end training with 2 layer graph transformer multitask
# python main.py \
#     --num_classes 9 \
#     --batch_size 4 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --fully_connected \
#     --is_global_feat \
#     --cls \
#     --image_featuriser efficientnet \
#     --task graph_transformer \
#     --run efficientnet_end_to_end_2L_cls_multitask \
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
#     --is_global_feat \
#     --cls \
#     --image_featuriser efficientnet \
#     --task graph_transformer \
#     --run efficientnet_end_to_end_2L_cls_multitask \
#     --num_workers 32

# # densenet end to end training with 1 layer graph transformer no global features image only classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 4 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run densenet_end_to_end_1L_no_global_feat_image_lvl \
#     --image_featuriser densenet \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 4 \
#     --lr 0.0001 \
#     --grad_accum 32 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run densenet_end_to_end_1L_no_global_feat_image_lvl \
#     --image_featuriser densenet \
#     --num_workers 32

# # 1 layer fully connected graph transformer
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_1_layer \
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
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_1_layer \
#     --num_workers 32

# # 1 layer fully connected graph transformer global image level classification no global features
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_1_layer_no_global_feat_image_only_classification \
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
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_1_layer_no_global_feat_image_only_classification \
#     --num_workers 32

# xfactor mean pool nodes, global image level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --pool mean \
#     --minimalistic \
#     --task xfactor \
#     --run mean_pool_global_image_classification \
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
#     --num_layers 1 \
#     --graph_importance 1.0 \
#     --pool mean \
#     --minimalistic \
#     --task xfactor \
#     --run mean_pool_global_image_classification \
#     --num_workers 32

# # fully connected graph transformer
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc \
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
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc \
#     --num_workers 32

# # fully connected graph transformer with pruned resnet50
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_pruned \
#     --prune \
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
#     --fully_connected \
#     --task graph_transformer \
#     --run gt_fc_pruned \
#     --prune \
#     --num_workers 32

# # graph transformer with learnt embeddings
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --abs_pos \
#     --task graph_transformer \
#     --run gt_abs_pos \
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
#     --abs_pos \
#     --task graph_transformer \
#     --run gt_abs_pos \
#     --num_workers 32

# # resnet50 final fc finetune
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --grad_accum 4 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --num_workers 32

# # graph transformer with one_cycle lr scheduler
# python main.py \
#     --epochs 20 \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --lr_scheduler one_cycle \
#     --task graph_transformer \
#     --run gt_one_cycle \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
# python temp_metrics_anaxnet.py \
#     --epochs 20 \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --lr_scheduler one_cycle \
#     --task graph_transformer \
#     --run gt_one_cycle \
#     --num_workers 32

# # ablation study
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --ablation \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers \
#     --num_workers 32

# # xfactor scene graph only, mean pool
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --pool mean \
#     --minimalistic \
#     --task xfactor \
#     --run scene_graph \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # graph transformer
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task graph_transformer \
#     --run graph_transformer_6_layers \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # graph_benchmark
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.0 \
#     --is_global_feat \
#     --task graph_benchmark \
#     --run graph_benchmark_node_only \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # xfactor contrastive
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --contrastive \
#     --hash_bits 1024 \
#     --task xfactor \
#     --run xfactor_6_layers_contrastive_importance \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # xfactor contrastive
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --contrastive \
#     --hash_bits 1024 \
#     --task xfactor \
#     --run xfactor_6_layers_contrastive \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # xfactor pool
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --pool mean \
#     --task xfactor \
#     --run xfactor_6_layers_mean_pool \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # graph_benchmark
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task graph_benchmark \
#     --run graph_benchmark \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # xfactor
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task xfactor \
#     --run xfactor_6_layers \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet large attn graph and node level classification with resiudal connections
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers_residual \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # ATH
# python main.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --gpu_ids 0 \
#     --num_workers 36 \
#     --train \
#     --log
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36

# # anaxnet contrastive large attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --contrastive \
#     --hash_bits 1024 \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_contrastive_with_importance \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet contrastive matryoshka large attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --contrastive \
#     --hash_bits 1024 \
#     --matryoshka \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_contrastive_matryoshka_with_importance \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet matryoshka large attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --matryoshka \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_matryoshka \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # contrastive anaxnet attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --hash_bits 64 \
#     --batch_size 8 \
#     --lr 0.0001 \
#     --grad_accum 8 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task contrastive \
#     --run contrastive_2_layers \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet large attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet large attn graph and node level classification with dropout
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers_dropout \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet attn graph and node level classification
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_complete \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet attn graph only
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 1.0 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_graph_only \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log

# # anaxnet attn node only
# python main.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 0.0 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_node_only \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log


# # fast anaxnet attn graph and node level classification
# # doesnt work though
# python main.py \
#     --num_classes 9 \
#     --batch_size 32 \
#     --lr 0.0001 \
#     --grad_accum 2 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer_fast \
#     --run fast \
#     --gpu_ids 0 1 2 3 \
#     --num_workers 10 \
#     --train \
#     --log
