#!/bin/bash

# Create the ckpts folder and download the checkpoint file
cd utils
mkdir ckpts

# Create subfolders for each checkpoint file 
# (download the checkpoint files from result table and place them in the corresponding folders)
cd ckpts
mkdir cifar10
mkdir cifar100
mkdir imagenet
mkdir pacs

cd ..

# Create the results folder
mkdir results

# Create subfolders for each dataset
# (download the datasets from the official sources (step 4) and place them in the corresponding folders)
mkdir data

cd data
mkdir CIFAR-10-C
mkdir CIFAR-100-C
mkdir Imagenet-C
mkdir Imagenet-S
mkdir Imagenet-R
mkdir Imagenet-V2
mkdir PACS

cd ../..

echo "Setup complete."
