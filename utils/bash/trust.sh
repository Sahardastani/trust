#!/bin/bash

# Define the parameters
GPU_ID=0
EXPERIMENT="traverse_permutation"
ADAPTED_PARAMETER="ss2d"
LOSS_CHOICE="pseudo_labeling"
BACKBONE='vmamba'

STEPS=1
LR=1e-4

ADAPT=True
EPISODIC=False
MODE_VARIATION="sequential"
MODEL_MERGING="weight_averaging"

# List of datasets
DATASETS=("imagenetc" "cifar10c" "cifar100c" "imagenetsketch" "imagenetv2" "imagenetr" "pacs")

# List of weights
WEIGHT_LIST="test_traverses_0123 \
             test_traverses_0132 \
             test_traverses_0321 \
             test_traverses_1023 \
             test_traverses_1032 \
             test_traverses_3120"

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
        --mode_variation $MODE_VARIATION \
        --model_merging $MODEL_MERGING \
        --weight_list $WEIGHT_LIST \
        --corruptions_list $CORRUPTIONS_LIST
done
