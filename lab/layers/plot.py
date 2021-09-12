from collections import OrderedDict

import joypy
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import numpy as np
import pandas as pd
from scipy.stats.stats import mode
import torch
import torch.nn as nn
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)
from lab.layers.resnet10 import ResNet10


def load_checkpoint(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    state = torch.load(load_path, map_location=torch.device(device))['state']
    clf_state = OrderedDict()
    state_keys = list(state.keys())
    for _, key in enumerate(state_keys):
        if "feature." in key:
            # an architecture model has attribute 'feature', load architecture
            # feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        elif "classifier." in key:
            newkey = key.replace("classifier.", "")
            clf_state[newkey] = state.pop(key)
        else:
            state.pop(key)
    model.load_state_dict(state)
    model.eval()
    return model


def load_checkpoint2(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    model.eval()
    return model


def get_BN_output(model, layers=None):
    colors = ['#990033', '#FF6699']  # first color is black, last is red
    newcolors = []
    labels = []
    flat_list = []

    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if (layers is None) or (i in layers):
                
                out = (layer.output.permute(
                    1, 0, 2, 3).mean([2, 3])).tolist()

                labels += ['Layer {}'.format(i+1)]+[None]*(len(out)-1)
                clm = LinearSegmentedColormap.from_list(
                    "Custom", colors, N=len(out))
                temp = clm(range(0, len(out)))
                for c in temp:
                    newcolors.append(c)

                for item in out:
                    flat_list.append(item)
            i += 1
    return flat_list, labels, ListedColormap(newcolors, name='OrangeBlue')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

model_names = ['Baseline','BMS_Eurosat','AdaBN_EuroSAT']
models = []
models.append(load_checkpoint(ResNet10(), 'logs/AdaBN/teacher_miniImageNet/399.tar', device))
models.append(load_checkpoint2(ResNet10(), 'logs/vanilla/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(ResNet10(), 'logs/AdaBN/EuroSAT/checkpoint_best.pkl', device))


b_size = 16
transform = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
split = 'datasets/split_seed_1/EuroSAT_unlabeled_20.csv'
dataset = EuroSAT_few_shot.SimpleDataset(
    transform, split=split)
EuroSAT_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                             num_workers=0,
                                             shuffle=True, drop_last=True)

transform = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
dataset = miniImageNet_few_shot.SimpleDataset(
    transform, split=None)
base_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                          num_workers=0,
                                          shuffle=True, drop_last=True)

EuroSAT_x, _ = iter(EuroSAT_loader).next()
base_x, _ = iter(base_loader).next()

# baseline_model(base_x)
# baseline_BN_base_out = get_BN_output(baseline_model)
# baseline_model(EuroSAT_x)
# baseline_BN_EuroSAT_out = get_BN_output(baseline_model)

# EuroSAT_model(base_x)
# EuroSAT_BN_base_out = get_BN_output(EuroSAT_model)
# EuroSAT_model(EuroSAT_x)
# EuroSAT_BN_EuroSAT_out = get_BN_output(EuroSAT_model)

# The EuroSAT

layers=None
for i, model in enumerate(models):
    model(EuroSAT_x)
    out, labels, clm = get_BN_output(model, layers=layers)
    joypy.joyplot(out, labels=labels, overlap=2, grid=True,
                colormap=clm, linecolor='w', linewidth=0.2, x_range=(-2,2))
    # plt.show()
    plt.savefig("{}_to_Eurosat.pdf".format(model_names[i]))
    print(i)
    model(base_x)
    out, labels, clm = get_BN_output(model, layers=layers)
    joypy.joyplot(out, labels=labels, overlap=2, grid=True,
                colormap=clm, linecolor='w', linewidth=0.2, x_range=(-2,2))
    # plt.show()
    plt.savefig("{}_to_MiniImageNet.pdf".format(model_names[i]))
    print(i)
    