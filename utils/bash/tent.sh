#!/bin/bash

# Define the parameters
GPU_ID=0
EXPERIMENT="tent"
ADAPTED_PARAMETER="bn"
LOSS_CHOICE="softmax_entropy"
BACKBONE='vmamba'
STEPS=1
LR=1e-4
ADAPT=True
EPISODIC=False

# List of datasets
DATASETS=("imagenetc" "cifar10c" "cifar100c" "imagenetsketch" "imagenetv2" "imagenetr" "pacs")

# Corruptions list
CORRUPTIONS_LIST="gaussian_noise \
                  shot_noise \
                  impulse_noise \
                  defocus_blur \
                  glass_blur \
                  motion_blur \
                  zoom_blur \
                  frost \
                  snow \
                  fog \
                  brightness \
                  contrast \
                  elastic_transform \
                  pixelate \
                  jpeg_compression"

# Iterate over each dataset
for DATASET in "${DATASETS[@]}"
do
    echo "Running experiment for dataset: $DATASET"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main.py \
        --experiment_name $EXPERIMENT \
        --adapted_parameter $ADAPTED_PARAMETER \
        --loss_choice $LOSS_CHOICE \
        --dataset $DATASET \
        --backbone $BACKBONE \
        --steps $STEPS \
        --lr $LR \
        --adapt $ADAPT \
        --episodic $EPISODIC \
        --corruptions_list $CORRUPTIONS_LIST
done
