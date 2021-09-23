from collections import OrderedDict
import statistics
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

from PIL import Image


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


def get_BN_output(model, colors, layers=None):
    newcolors = []
    labels = []
    BN_list = []
    if layers is None:
        flatten = True
    else: 
        flatten = False

    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if (layers is None) or (i in layers):
                flat_list = []
                flat_list = layer.weight.tolist()

                # for channel in out:
                #     if flatten:
                #         flat_list += [channel]
                #     else:
                #         flat_list.append(channel)

                if flatten:
                    BN_list.append(flat_list)
                    labels += ['Layer {0:02d} ({1: 0.2f}, {2: 0.2f})'.format(
                        i+1, statistics.mean(flat_list), statistics.stdev(flat_list))]
                else:
                    BN_list += flat_list
                    labels += ['Layer {0:02d}'.format(i+1)]
                    labels += [None]*(len(out)-1)

                    clm = LinearSegmentedColormap.from_list(
                        "Custom", colors, N=len(out))
                    temp = clm(range(0, len(out)))
                    for c in temp:
                        newcolors.append(c)

            i += 1
    if flatten:
        clm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=i)
        temp = clm(range(0, i))
        for c in temp:
            newcolors.append(c)

    return BN_list, labels, ListedColormap(newcolors, name='custom')


def to_grid(path_list, out = "lab/layers/grid.png"):
    
    blank_image = None
    top_left = (0, 0)
    for i in range(0, len(path_list), 2):
        left = Image.open(path_list[i], mode='r')
        right = Image.open(path_list[i+1], mode='r')
        if blank_image is None:
            blank_image = Image.new(mode='RGB', size=(left.width * 2, right.height * 4 ))
        blank_image.paste(left, top_left)
        top_left = (top_left[0] + left.width, top_left[1])
        blank_image.paste(right, top_left)
        top_left = (0, top_left[1] + right.height)
    blank_image.save(out)


device = torch.device("cpu")

model_names = ['Baseline', 'BMS_Eurosat', 'AdaBN_EuroSAT', 'STARTUP_EuroSAT']
models = []
models.append(load_checkpoint2(
    ResNet10(), 'logs/baseline/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/vanilla/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/AdaBN/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/STARTUP/EuroSAT/checkpoint_best.pkl', device))


b_size = 32
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



colors = [['#670022', '#FF6699'], ['#004668', '#66D2FF'],
          ['#9B2802', '#FF9966'], ['#346600', '#75E600']]

l = None
path_list = []
for i, model in enumerate(models):
        with torch.no_grad():
            model(base_x)
            mini_out, mini_labels, clm = get_BN_output(
                model, colors=colors[i], layers=l)
            
            model(EuroSAT_x)
            Euro_out, EuroSAT_labels, clm = get_BN_output(
                model, colors=colors[i], layers=l)
            
            args = {'overlap': 4, 'bw_method': 0.2,
                    'colormap': clm, 'linewidth': 0.3, 'x_range': [-2, 2], 'linecolor': 'w',
                    'background': 'w',  'alpha': 0.8, 'figsize': (10, 5), 'fill': True,
                    'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(base_x))}

            joypy.joyplot(list(reversed(mini_out)), labels= list(reversed(mini_labels)), **args)
            # plt.show()
            path_list.append(
                "./lab/affines/{0}_to_MiniImageNet.png".format(model_names[i]))
            plt.savefig(path_list[-1],)

            joypy.joyplot(list(reversed(Euro_out)), labels= list(reversed(EuroSAT_labels)), **args)
            # plt.show()
            path_list.append(
                "./lab/affines/{0}_to_EuroSAT.png".format(model_names[i]))
            plt.savefig(path_list[-1],)
        to_grid(path_list, out = "lab/affines/grid_all.png")
    # plt.cla()
