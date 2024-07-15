#!/bin/bash
#SBATCH -A chocolite
#SBATCH -J mimic-cxr-emb
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3G
#SBATCH --constraint=2080ti
#SBATCH --time=4-00:00:00

mkdir -p /ssd_scratch/cvit/arihanth/physionet.org/files
mkdir -p /ssd_scratch/cvit/arihanth/checkpoints
cp ~/arihanth/data/emb.zip /ssd_scratch/cvit/arihanth/physionet.org/files/
unzip /ssd_scratch/cvit/arihanth/physionet.org/files/emb.zip -d /ssd_scratch/cvit/arihanth/physionet.org/files/ -y

find /ssd_scratch/cvit/arihanth -type f -exec touch {} +

# for embeddings only
python main.py \
    --emb_dim 1376 \
    --num_classes 14 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 100 \
    --grad_accum 1 \
    --task mimic-cxr-emb \
    --run emb \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --compile \
    --log

# for images
python main.py \
    --emb_dim 1000 \
    --num_classes 14 \
    --batch_size 48 \
    --lr 0.0005 \
    --grad_accum 1 \
    --task mimic-cxr-jpg \
    --run jpg_refactored \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 32 \
    --train \
    --log

# for graphs on images
python main.py \
    --emb_dim 1024 \
    --edge_dim 32 \
    --num_layers 2 \
    --num_classes 14 \
    --batch_size 8 \
    --lr 0.0006 \
    --grad_accum 8 \
    --task graph-jpg \
    --run graph_jpg \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 16 \
    --train \
    --log

# gcn
python main.py \
    --emb_dim 1024 \
    --edge_dim 32 \
    --num_layers 2 \
    --num_classes 14 \
    --batch_size 8 \
    --lr 0.0006 \
    --grad_accum 8 \
    --task gcn \
    --run gcn \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 16 \
    --train \
    --log

# gcn preprocessed
python main.py \
    --emb_dim 1024 \
    --edge_dim 32 \
    --num_layers 2 \
    --num_classes 14 \
    --batch_size 16 \
    --lr 0.0006 \
    --grad_accum 1 \
    --task gcn_preprocessed \
    --run gcn_preprocessed \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 16 \
    --train \
    --log

# graph transformer on precomputed features
python main.py \
    --emb_dim 1024 \
    --edge_dim 32 \
    --num_layers 2 \
    --num_classes 14 \
    --batch_size 128 \
    --lr 0.0006 \
    --grad_accum 1 \
    --task graph_transformer \
    --run graph_transformer \
    --validate_every -1 \
    --validate_for -1 \
    --gpu_ids 0 1 2 3 \
    --num_workers 16 \
    --train \
    --log