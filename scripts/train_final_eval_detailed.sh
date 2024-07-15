#!/bin/bash
#SBATCH -A arihanth.srikar
#SBATCH -J naren_eval077
#SBATCH --reservation naren_MICCAI
#SBATCH --output=/tmp/naren_eval077.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH -w gnode077
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

# best config with learnt embeddings and edges: graph transformer - Naren 20-40 age
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
    --num_workers 32 \
    --naren \
    --lower_age_limit 20 \
    --upper_age_limit 40

# resnet50 - Naren 20-40 age
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc_new \
    --num_workers 32 \
    --naren \
    --lower_age_limit 20 \
    --upper_age_limit 40

# anaxnet - Naren 20-40 age
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 32 \
    --naren \
    --lower_age_limit 20 \
    --upper_age_limit 40

# ath - Naren 20-40 age
python temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36 \
    --naren \
    --lower_age_limit 20 \
    --upper_age_limit 40

# best config with learnt embeddings and edges: graph transformer - Naren 40-60 age
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
    --num_workers 32 \
    --naren \
    --lower_age_limit 40 \
    --upper_age_limit 60

# resnet50 - Naren 40-60 age
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc_new \
    --num_workers 32 \
    --naren \
    --lower_age_limit 40 \
    --upper_age_limit 60

# anaxnet - Naren 40-60 age
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 32 \
    --naren \
    --lower_age_limit 40 \
    --upper_age_limit 60

# ath - Naren 40-60 age
python temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36 \
    --naren \
    --lower_age_limit 40 \
    --upper_age_limit 60

# best config with learnt embeddings and edges: graph transformer - Naren 60-80 age
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
    --num_workers 32 \
    --naren \
    --lower_age_limit 60 \
    --upper_age_limit 80

# resnet50 - Naren 60-80 age
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc_new \
    --num_workers 32 \
    --naren \
    --lower_age_limit 60 \
    --upper_age_limit 80

# anaxnet - Naren 60-80 age
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 32 \
    --naren \
    --lower_age_limit 60 \
    --upper_age_limit 80

# ath - Naren 60-80 age
python temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36 \
    --naren \
    --lower_age_limit 60 \
    --upper_age_limit 80

# best config with learnt embeddings and edges: graph transformer - Naren 80-100 age
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
    --num_workers 32 \
    --naren \
    --lower_age_limit 80 \
    --upper_age_limit 100

# resnet50 - Naren 80-100 age
python temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc_new \
    --num_workers 32 \
    --naren \
    --lower_age_limit 80 \
    --upper_age_limit 100

# anaxnet - Naren 80-100 age
python temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet \
    --num_workers 32 \
    --naren \
    --lower_age_limit 80 \
    --upper_age_limit 100

# ath - Naren 80-100 age
python temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36 \
    --naren \
    --lower_age_limit 80 \
    --upper_age_limit 100



# # best config with learnt embeddings and edges: graph transformer - Naren gender male
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
#     --num_workers 32 \
#     --naren \
#     --gender M

# # resnet50 - Naren gender male
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --num_workers 32 \
#     --naren \
#     --gender M

# # anaxnet - Naren gender male
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --task anaxnet \
#     --run anaxnet \
#     --num_workers 32 \
#     --naren \
#     --gender M

# # ath - Naren gender male
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36 \
#     --naren \
#     --gender M

# # best config with learnt embeddings and edges: graph transformer - Naren gender female
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
#     --num_workers 32 \
#     --naren \
#     --gender F

# # resnet50 - Naren gender female
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --num_workers 32 \
#     --naren \
#     --gender F

# # anaxnet - Naren gender female
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --task anaxnet \
#     --run anaxnet \
#     --num_workers 32 \
#     --naren \
#     --gender F

# # ath - Naren gender female
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36 \
#     --naren \
#     --gender F

# # best config with learnt embeddings and edges: graph transformer - Naren AP
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
#     --num_workers 32 \
#     --naren \
#     --view AP

# # best config with learnt embeddings and edges: graph transformer - Naren PA
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
#     --num_workers 32 \
#     --naren \
#     --view PA





# # resnet50 - Naren gender AP
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --num_workers 32 \
#     --naren \
#     --view AP

# # resnet50 - Naren gender PA
# python temp_metrics_resnet50.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0005 \
#     --task resnet50 \
#     --run resnet50_fc_new \
#     --num_workers 32 \
#     --naren \
#     --view PA





# # anaxnet - Naren AP
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --task anaxnet \
#     --run anaxnet \
#     --num_workers 32 \
#     --naren \
#     --view AP

# # anaxnet - Naren PA
# python temp_metrics_anaxnet.py \
#     --num_classes 9 \
#     --batch_size 16 \
#     --lr 0.0001 \
#     --task anaxnet \
#     --run anaxnet \
#     --num_workers 32 \
#     --naren \
#     --view PA





# # ath - Naren AP
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36 \
#     --naren \
#     --view AP

# # ath - Naren PA
# python temp_metrics_ath.py \
#     --num_classes 9 \
#     --batch_size 24 \
#     --lr 0.001 \
#     --dropout 0.0 \
#     --hash_bits 32 \
#     --task ath \
#     --run ath \
#     --num_workers 36 \
#     --naren \
#     --view PA
