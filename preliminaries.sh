#!/bin/bash

# Create the ckpts folder and download the checkpoint file
cd utils
mkdir ckpts

cd ckpts
mkdir imagenet
mkdir cifar10
mkdir cifar100
mkdir visdac
mkdir pacs

cd imagenet
cp -r /data/shared/vmamba_ckpts/* ./

cd ../cifar10
cp -r /data/shared/vmamba_ckpts/cifar10/* ./

cd ../cifar100
cp -r /data/shared/vmamba_ckpts/cifar100/* ./

cd ../pacs
cp -r /data/shared/vmamba_ckpts/pacs/* ./

cd ../..

# Create the results folder
mkdir results

# Create the data folder and create a symbolic link to Imagenet-C
mkdir data
cd data
ln -s /data/shared/Imagenet-C/
ln -s /data/shared/Imagenet-S/
ln -s /data/shared/Imagenet-R/
ln -s /data/shared/Imagenet-V2/
ln -s /data/shared/CIFAR-10-C/
ln -s /data/shared/CIFAR-100-C/
ln -s /data/shared/PACS/
cd ../..

echo "Setup complete."
