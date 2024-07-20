
<h3 align="center">CheXtriev: Anatomy-Centered Representation for Case-Based Retrieval of Chest Radiographs</h3>

<h4 align="center"><a href="https://narenakash.github.io/">Naren Akash R J</a>&dagger;, <a href="https://arihanth007.github.io/porfolio/about.html">Arihanth Srikar</a>&dagger;, <a href="https://cvit.iiit.ac.in/mip/">Jayanthi Sivaswamy</a></h4>

<p align="center">
  <a href="#arxiv">ArXiv</a> •
  <a href="#installation">Installation</a> •
  <a href="#chextriev">Training</a> •
  <a href="#citing">Cite</a>
</p>

> [!NOTE]  
> 🎉 CheXtriev has been accepted at [MICCAI 2024](https://conferences.miccai.org/2024/en/)!


## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Baselines](#baselines)
- [CheXtriev Variants](#chextriev)
- [Citation]()
  

## Installation
To set up the environment, run the following command to create a conda environment:
```bash
conda env create -f environment.yml
```
The main dependencies required are Pytorch, Pytorch Lightning, Pytorch Geometric and Faiss.

## Project Structure
- Run `main.py` to train any model. Modify paths to the dataset and model checkpoints as necessary.
- `metrics` contains evaluation scripts for various models. These scripts utilize `common_metrics.py` to compute metrics such as mAP (Mean Average Precision), mHR (Mean Hit Rate), and mRR (Mean Reciprocal Rank). Modify paths to the dataset and model checkpoints within these scripts as needed.
- `dataloader` contains the dataloader implementations for each model in the respective formats.
- `graph_transformer` is adapted from the well maintained [GitHub repository](https://github.com/lucidrains/graph-transformer-pytorch) of the graph transformer architecture with added functionalities to support the project requirements.
- `model` contains definitions and architectures for the various models used in the project.
- `notebooks` includes Jupyter notebooks used for analysis, visualizations, and initial experiments. These were later converted to Python scripts for streamlined execution.
- `others` contains scripts for data processing and transferring data to HPC cluster, specific to our setup.
- `output` is where results are stored in a tabular format, detailing top-3, top-5, and top-10 retrieved images.
- `Res2Net` contains the multi-scale ResNet50 model definition borrowed from [this repository](https://github.com/Res2Net/Res2Net-PretrainedModels).
- `scripts` includes the command scripts to train any model including hyperparameter tuning.

## Baselines

### 1. Global CNN
The Global CNN baseline utilizes ResNet50 to extract latent representations from chest radiographs. Only the classification head is finetuned, while the rest of the network's weights are frozen.

#### Training
Use the following script to train the ResNet50 model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --grad_accum 4 \
    --task resnet50 \
    --run resnet50_fc \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the ResNet50 model:
```python
python metrics/temp_metrics_resnet50.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0005 \
    --task resnet50 \
    --run resnet50_fc \
    --num_workers 32
```

### 2. ATH
Attention-based Triplet Hashing (ATH) is a state-of-the-art chest radiograph retrieval method based on attention mechanism and triplet hashing. More details can be found in the [Github repository](https://github.com/fjssharpsword/ATH) and the [paper](https://arxiv.org/pdf/2101.12346).

#### Training
Use the following script to train the ATH model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --gpu_ids 0 \
    --num_workers 36 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the ATH model:
```python
python metrics/temp_metrics_ath.py \
    --num_classes 9 \
    --batch_size 24 \
    --lr 0.001 \
    --grad_accum 4 \
    --dropout 0.0 \
    --hash_bits 32 \
    --task ath \
    --run ath \
    --num_workers 36
```

## 3. AnaXNet
AnaXNet is an anatomy-aware multi-label classification model for chest X-rays. For more details, refer to the [paper](https://miccai2021.org/openaccess/paperlinks/2021/09/01/053-Paper1467.html).

#### Training
Use the following script to train the AnaXNet model:
```python
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
```

#### Evaluation
Use the following script to evaluate the AnaXNet model:
```python
python metrics/temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --task anaxnet \
    --run anaxnet_final \
    --num_workers 32
```

## CheXtriev
CheXtriev is a novel graph-based, anatomy-aware framework designed for chest radiograph retrieval. It consists of several variants (V0 to V6), each incorporating various enhancements and modifications.

### V0
This variant extracts ResNet50 features from the predefined 18 anatomical regions, and uses mean pooling to obtain the latent representation of the chest radiographs.
<!-- xfactor mean pool nodes, global image level classification -->
#### Global Image Level Classification Training
Use the following script to train the V0 model for global image level classification:
```python
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
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```
#### Global Image Level Classification Evaluation
Use the following script to evaluate the V0 model for global image level classification:
```python
python metrics/temp_metrics_anaxnet.py \
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
```

<!-- xfactor mean pool nodes, node level classification -->
#### Local Anatomy Level Classification Training
Use the following script to train the V0 model for local anatomy level classification:
```python
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
```

#### Local Anatomy Level Classification Evaluation
Use the following script to train the V0 model for local anatomy level classification:
```python
python metrics/temp_metrics_anaxnet.py \
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
```

### V1
In V1, anatomical features processed through ResNet50 are further contextualized using a graph transformer, with edge connections (binary) based on label co-occurence. This model is supervised globally at the image level.


#### Training
Use the following script to train the V1 model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --task graph_transformer \
    --run best_config_adj_mat \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the V1 model:
```python
python metrics/temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --task graph_transformer \
    --run best_config_adj_mat \
    --num_workers 32
```

### V2
In V2, anatomical features processed through ResNet50 are further contextualized using a graph transformer, with fully connected uniform edge connections to model relationships among the anatomical structures. This model is supervised globally at the image level.

#### Training
Use the following script to train the V2 model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --task graph_transformer \
    --run best_config_abs_pos \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the V2 model:
```python
python metrics/temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --task graph_transformer \
    --run best_config_abs_pos \
    --num_workers 32
```

### V3
V3 builds on V2 by introducing learnable positional embeddings, enhancing the model's ability to capture spatial relationships between anatomical features.

#### Training
Use the following script to train the V3 model:
```python
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
    --task graph_transformer \
    --run best_config_abs_pos \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the V3 model:
```python
python metrics/temp_metrics_anaxnet.py \
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
```

### V4
V4 modifies V3 by making the fully connected edges unique and entirely learnable and supervised globally at the image level. We use local multi-level features with gated residuals in V4 only.

#### Training
Use the following script to train the V4 model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 0.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --residual_type 2 \
    --task graph_transformer \
    --run best_config_with_edges_local_anatomy \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the V4 model:
```python
python metrics/temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 0.0 \
    --fully_connected \
    --abs_pos \
    --accept_edges \
    --residual_type 2 \
    --task graph_transformer \
    --run best_config_with_edges_local_anatomy \
    --num_workers 32
```

### V5
V5 alters V4 by omitting the learnable positional embeddings, supervising globally at the image level and uses global multi-level features with gated residuals.

#### Training
Use the following script to train the V5 model:
```python
python main.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --grad_accum 8 \
    --dropout 0.0 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --accept_edges \
    --residual_type 2 \
    --task graph_transformer \
    --run best_config_with_edges_without_pos_emb \
    --gpu_ids 0 1 \
    --num_workers 20 \
    --train \
    --log
```

#### Evaluation
Use the following script to evaluate the V5 model:
```python
python metrics/temp_metrics_anaxnet.py \
    --num_classes 9 \
    --batch_size 16 \
    --lr 0.0001 \
    --num_layers 2 \
    --graph_importance 1.0 \
    --fully_connected \
    --accept_edges \
    --residual_type 2 \
    --task graph_transformer \
    --run best_config_with_edges_without_pos_emb \
    --num_workers 32
```

### V6
V6 is the best configuration, where detected anatomies are processed through ResNet50 and then passed through two layers of Graph Transformers with learnable continuous edges and positional embeddings. This model is supervised globally at the image level.

#### Training
Use the following script to train the V6 model:
```python
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
```

#### Evaluation
Use the following script to evaluate the V6 model:
```python
python metrics/temp_metrics_anaxnet.py \
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
```
