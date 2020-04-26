# gan test
import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.gan_dataset import *
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents

# gan = True
# if gan:
# 	# get model
discriminator_path = 'model_file/Discriminator_cpu_50.pt'
generator_path = 'model_file/Generator_cpu_50.pt'


GAN = CGAN_(discriminator_path, generator_path)
label = GAN.generate_label(10)
img = GAN.generate_image(label)

# print(label)
# print(target)
normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))
val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])

dataset = GAN_MNIST(img, label, transform=val_transform)
# img, tgt = dataset[5]

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True)

train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                     args.n_permutation,
                                                                     remap_class=not args.no_class_remap)