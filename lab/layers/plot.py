from collections import OrderedDict

import joypy
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)
from lab.layers.resnet10 import ResNet10
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats.stats import mode


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


def get_BN_output(model, colors, layers=None, flatten=False):
    newcolors = []
    labels = []
    channel_list = []

    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if (layers is None) or (i in layers):
                flat_list = []
                out = layer.output / \
                    layer.weight[None, :, None, None] - \
                    layer.bias[None, :, None, None]
                out = (layer.output.permute(
                    1, 0, 2, 3).mean([2, 3])).tolist()

                if flatten:
                    labels += ['Layer {0:02d}'.format(i+1)]
                else:
                    labels += ['Layer {0:02d}'.format(i+1)]+[None]*(len(out)-1)

                    clm = LinearSegmentedColormap.from_list(
                        "Custom", colors, N=len(out))
                    temp = clm(range(0, len(out)))
                    for c in temp:
                        newcolors.append(c)

                for channel in out:
                    if flatten:
                        flat_list += channel
                    else:
                        flat_list.append(channel)
                if flatten:
                    channel_list.append(flat_list)
                else:
                    channel_list += flat_list

            i += 1
    if flatten:
        clm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=i)
        temp = clm(range(0, i))
        for c in temp:
            newcolors.append(c)

    return channel_list, labels, ListedColormap(newcolors, name='OrangeBlue')


device = torch.device("cpu")

model_names = ['Baseline', 'BMS_Eurosat', 'AdaBN_EuroSAT', 'STARTUP_EuroSAT']
models = []
models.append(load_checkpoint(
    ResNet10(), 'logs/AdaBN/teacher_miniImageNet/399.tar', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/vanilla/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/AdaBN/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/STARTUP/EuroSAT/checkpoint_best.pkl', device))


b_size = 128
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


for l in [1]:  # range(12):
    layers = range(12)
    colors = [['#670022', '#FF6699'], ['#004668', '#66D2FF'],
              ['#9B2802', '#FF9966'], ['#346600', '#75E600']]
    for i, model in enumerate(models):
        with torch.no_grad():
            model(EuroSAT_x)
            mini_out, labels, clm = get_BN_output(
                model, colors=colors[i], layers=layers, flatten=True)
            model(base_x)
            Euro_out, labels, clm = get_BN_output(
                model, colors=colors[i], layers=layers, flatten=True)

            args = {'labels': list(reversed(labels)), 'overlap': 2,
                    'colormap': clm, 'linecolor': 'black', 'linewidth': 0.0,
                    'background': 'w', 'x_range':[-2, 2],
                    'grid': True, 'hist': False, 'bins': int(len(base_x)/4)}

            joypy.joyplot(list(reversed(mini_out)), **args)
            # plt.show()
            plt.savefig(
                "./lab/layers/{0}_to_EuroSAT_layer{1}.pdf".format(model_names[i], l))
            print(i)

            joypy.joyplot(list(reversed(Euro_out)), **args)
            # plt.show()
            plt.savefig(
                "./lab/layers/{0}_to_MiniImageNet_layer{1}.pdf".format(model_names[i], l))
            print(i)
