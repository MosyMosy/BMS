from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from openTSNE import TSNE
from seaborn.palettes import color_palette


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

with torch.no_grad():
    bms_feature = torch.load('t-sne_bms_feature.pt')
    base_features = torch.load('t-sne_base_features.pt')
    label_dataset = torch.load('t-sne_label_dataset.pt')

    base_embedding = TSNE().fit(base_features.numpy())
    bms_embedding = TSNE().fit(bms_feature.numpy())
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    color = sns.color_palette(n_colors=len(5))
    sns.kdeplot(x=base_embedding[:, 0], y=base_embedding[:, 1],
                hue=label_dataset, ax=ax[0], palette=color).set(title='baseline')
    sns.kdeplot(x=bms_embedding[:, 0], y=bms_embedding[:, 1],
                hue=label_dataset, ax=ax[1], palette=color).set(title='BMS')
    plt.savefig("t-sne_kde.pdf")

    plt.show()
