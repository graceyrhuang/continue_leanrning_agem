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
current_dir = os.getcwd()
discriminator_path = os.path.join(current_dir, 'model_file/Discriminator_cpu_50.pt')
generator_path = os.path.join(current_dir, 'model_file/Generator_cpu_50.pt')


GAN = CGAN_(discriminator_path, generator_path)
label = GAN.generate_label(10)
img = GAN.generate_image(label)

normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))
val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])

dataset = GAN_MNIST(img, label, transform=val_transform)
print(dataset.number_classes)
# img, tgt = dataset[5]

