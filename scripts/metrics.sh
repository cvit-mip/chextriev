#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J evaluate
#SBATCH --reservation arihanth_MIDL
#SBATCH --output=logs/train_densenet.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode046
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --num_layers 6 \
    --graph_importance 0.2 \
    --is_global_feat \
    --ablation \
    --task anaxnet_attn_multilayer \
    --run anaxnet_attn_6_layers \
    --num_workers 32

# python process_data_dump.py \
#     --num_classes 9 \
#     --batch_size 32 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers \
#     --num_workers 32

# xfactor scene graph only, mean pool
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --num_layers 6 \
    --graph_importance 0.2 \
    --is_global_feat \
    --pool mean \
    --minimalistic \
    --task xfactor \
    --run scene_graph \
    --num_workers 32

# graph transformer
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --num_layers 6 \
    --graph_importance 0.2 \
    --is_global_feat \
    --task graph_transformer \
    --run graph_transformer_6_layers \
    --num_workers 32

# # graph_benchmark
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 2 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task graph_benchmark \
#     --run graph_benchmark_node_only \
#     --num_workers 32

# # xfactor contrastive
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # xfactor contrastive
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # xfactor pool
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # anaxnet original
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --task anaxnet \
#     --run anaxnet \
#     --num_workers 32

# # graph_benchmark
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# xfactor
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # anaxnet large attn graph and node level classification with resiudal connections
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # ATH
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36

# # anaxnet matryoshka large attn graph and node level classification
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # occlusion analysis on
# # anaxnet large attn graph and node level classification
# python analysis_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 32 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers \
#     --num_workers 32

# # occlusion analysis on
# # anaxnet large attn graph and node level classification without global features
# python analysis_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 32 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.0 \
#     --num_layers 6 \
#     --graph_importance 0.2 \
#     --task anaxnet_attn_multilayer \
#     --run anaxnet_attn_6_layers \
#     --num_workers 32

# # resnet50
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --grad_accum 4 \
#     --task resnet50 \
#     --run resnet50 \
#     --num_workers 32

# # densenet121
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --grad_accum 4 \
#     --task densenet \
#     --run densenet_9_classes \
#     --num_workers 32

# # anaxnet large attn graph and node level classification
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # anaxnet large attn graph and node level classification with dropout
# python temp_metrics_anaxnet.py \
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
#     --num_workers 32

# # anaxnet attn graph and node level classification
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 0.2 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_complete \
#     --num_workers 32

# # anaxnet attn graph only
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 1.0 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_graph_only \
#     --num_workers 32

# # anaxnet attn node only
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --grad_accum 4 \
#     --dropout 0.1 \
#     --graph_importance 0.0 \
#     --is_global_feat \
#     --task anaxnet_attn \
#     --run anaxnet_attn_node_only \
#     --num_workers 32
