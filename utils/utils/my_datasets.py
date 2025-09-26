import os
import json
import random
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode

imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

cifar_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def transform_fn(example):
    example["image"] = imagenet_transform(example["image"].convert("RGB"))  # Convert to RGB and apply transforms
    example["label"] = torch.tensor(example["label"], dtype=torch.long)  # Ensure label is tensor
    return example

def prepare_test_data(args, corruption, size=10000, transform=None):
    
    if args.dataset == 'imagenetc':
        teset = ImageFolder(os.path.join(args.data_dir, 'Imagenet-C', corruption, str(args.severity)), 
        transform=imagenet_transform)

    elif args.dataset == 'imagenetsketch':
        teset = ImageFolder(os.path.join(args.data_dir, 'Imagenet-S'), 
        transform=imagenet_transform)

    elif args.dataset == 'imagenetv2':
        teset = ImageFolder(os.path.join(args.data_dir, 'Imagenet-V2'), 
        transform=imagenet_transform)

    elif args.dataset == 'imagenetr':

        with open('utils/utils/imagenet_class_index.json', 'r') as f:
            raw_mapping = json.load(f)

        imagenet_class_to_idx = {v[0]: int(k) for k, v in raw_mapping.items()}
        dataset = ImageFolder(os.path.join(args.data_dir, 'Imagenet-R'))
        dataset.class_to_idx = {cls: imagenet_class_to_idx[cls] for cls in dataset.classes}

        def custom_target_transform(target):
            return dataset.class_to_idx[dataset.classes[target]]

        teset = ImageFolder(os.path.join(args.data_dir, 'Imagenet-R'),
                            transform=imagenet_transform,
                            target_transform=custom_target_transform)

    elif args.dataset == 'cifar10c':
        dataset_raw = np.load(args.data_dir + '/CIFAR-10-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(args.severity - 1)*size: args.severity*size]
        teset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR-10-C'), train=False, download=False, transform=cifar_transform)
        teset.data = dataset_raw

    elif args.dataset == 'cifar100c':
        dataset_raw = np.load(args.data_dir + '/CIFAR-100-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(args.severity - 1)*size: args.severity*size]
        teset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'CIFAR-100-C'), train=False, download=False, transform=cifar_transform)
        teset.data = dataset_raw

    elif args.dataset == 'pacs':
        teset = ImageFolder(os.path.join(args.data_dir, 'PACS', 'photo'), 
        transform=imagenet_transform)

    else:
        raise Exception('Dataset not found!')

    teloader = torch.utils.data.DataLoader(teset, 
                                           batch_size=args.batch_size, 
                                           shuffle=True,
                                           num_workers=args.workers, 
                                           sampler=None,  
                                           drop_last=False)

    return teloader, teset