from collections import OrderedDict

import matplotlib.pyplot as plt
# import seaborn as sns
import torch
# from openTSNE import TSNE
# from seaborn.palettes import color_palette

import models
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ImageNet_few_shot, ISIC_few_shot, miniImageNet_few_shot,
                      tiered_ImageNet_few_shot)


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


def load_checkpoint2(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])

    return sd['epoch']


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

dataset_class_list = [miniImageNet_few_shot, EuroSAT_few_shot,
                      CropDisease_few_shot, Chest_few_shot, ISIC_few_shot]
dataset_names_list = ['miniImageNet', 'EuroSAT',
                      'CropDisease', 'ChestX', 'ISIC']

# dataset_class_list = [miniImageNet_few_shot,
#                       EuroSAT_few_shot, CropDisease_few_shot]
# dataset_names_list = ['miniImageNet', 'EuroSAT',  'CropDisease']

dataloader_list = []
for i, dataset_class in enumerate(dataset_class_list):
    transform = dataset_class.TransformLoader(
        224).get_composed_transform(aug=True)
    transform_test = dataset_class.TransformLoader(
        224).get_composed_transform(aug=False)
    split = 'datasets/split_seed_1/{0}_unlabeled_20.csv'.format(
        dataset_names_list[i])
    if dataset_names_list[i] == 'miniImageNet':
        split = None
    dataset = dataset_class.SimpleDataset(
        transform, split=split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                         num_workers=0,
                                         shuffle=True, drop_last=True)
    dataloader_list.append(loader)

vanilla_models = []
baseline_model = models.ResNet10()
load_checkpoint(
    baseline_model, 'logs/AdaBN/teacher_miniImageNet/399.tar', device)
baseline_model.eval()
vanilla_models.append(baseline_model)

for name in dataset_names_list[1:]:
    vanilla_models.append(models.ResNet10())
    load_checkpoint2(vanilla_models[-1],
                    'logs/vanilla/{0}/checkpoint_best.pkl'.format(name), device)
    vanilla_models[-1].eval()




label_dataset = []
base_features = []
bms_feature = []
with torch.no_grad():
    for i, loader in enumerate(dataloader_list):
        # loader_iter = iter(loader)
        # x, _ = loader_iter.next()
        for x, _ in loader:
            base_features += baseline_model(x)
            bms_feature += vanilla_models[i](x)
            label_dataset += [dataset_names_list[i]]*len(x)
            # break

    base_features = torch.stack(base_features)
    bms_feature = torch.stack(bms_feature)
    
    torch.save(base_features, 't-sne_base_features.pt')
    torch.save(bms_feature, 't-sne_bms_feature.pt')

    # base_embedding = TSNE().fit(base_features.numpy())
    # bms_embedding = TSNE().fit(bms_feature.numpy())
    # fig, ax = plt.subplots(1, 2,figsize=(10,5))
    # color = sns.color_palette(n_colors=len(dataloader_list))
    # sns.kdeplot(x=base_embedding[:, 0], y=base_embedding[:, 1],
    #             hue=label_dataset, ax=ax[0], palette=color).set(title='baseline')
    # sns.kdeplot(x=bms_embedding[:, 0], y=bms_embedding[:, 1], 
    #             hue=label_dataset, ax=ax[1], palette=color).set(title='BMS')
    # plt.savefig("t-sne_kde.pdf")
    
    # plt.show()
