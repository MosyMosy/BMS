import random
import math
import copy
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from collections import OrderedDict
import warnings
import models
import time
import data
import utils
import sys
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.utils.data
from configs import miniImageNet_path, ISIC_path, ChestX_path, CropDisease_path, EuroSAT_path, ImageNet_test_path

torch.cuda.empty_cache()


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# import wandb

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x



def main(args):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    torch.cuda.empty_cache()
    # Set the scenes
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, time.strftime("%Y%m%d-%H%M%S") + '_checkpoint.log'), __name__)
    vallog = utils.savelog(args.dir, 'val')

    # wandb.init(project='STARTUP',
    #            group=__file__,
    #            name=f'{__file__}_{args.dir}')

    # wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create Models
    ###########################
    print("Loading Model: ", args.embedding_load_path)
    if args.embedding_load_path_version == 0:
        state = torch.load(args.embedding_load_path, map_location=torch.device(device))['state']
        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        sd = state
    elif args.embedding_load_path_version == 1:
        sd = torch.load(args.embedding_load_path,
                        map_location=torch.device(device))

        if 'epoch' in sd:
            print("Model checkpointed at epoch: ", sd['epoch'])
        sd = sd['model']
    
    else:
        raise ValueError("Invalid load path version!")

    if args.model == 'resnet10':
        pretrained_model = models.ResNet10()
        feature_dim = pretrained_model.final_feat_dim
    elif args.model == 'resnet12':
        pretrained_model = models.Resnet12(width=1, dropout=0.1)
        feature_dim = pretrained_model.output_size
    elif args.model == 'resnet18':
        pretrained_model = models.resnet18(remove_last_relu=False,
                                                    input_high_res=True)
        feature_dim = 512
    else:
        raise ValueError("Invalid model!")

    pretrained_model.load_state_dict(sd)
    clf = nn.Linear(feature_dim, 1000).to(device)

    ############################

    ###########################
    # Create DataLoader
    ###########################
    print(args.base_dataset)
    # create the base dataset
    if args.base_dataset == 'miniImageNet':
        base_transform = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        base_transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = miniImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'tiered_ImageNet':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        base_transform = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)
        if args.base_split is not None:
            base_dataset = tiered_ImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
    elif args.base_dataset == 'ImageNet_test':
        if args.base_no_color_jitter:
            base_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            warnings.warn("Using ImageNet with Color Jitter")
            base_transform = ImageNet_few_shot.TransformLoader(
                args.image_size).get_composed_transform(aug=True)
        base_transform_test = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(
            root=args.base_path, transform=base_transform)

        if args.base_split is not None:
            base_dataset = ImageNet_few_shot.construct_subset(
                base_dataset, args.base_split)
        print("Size of Base dataset:", len(base_dataset))
    else:
        raise ValueError("Invalid base dataset!")

    
    base_dataset_val = copy.deepcopy(base_dataset)
    base_dataset_val.transform = base_transform_test

    base_valset = base_dataset

    base_valloader = torch.utils.data.DataLoader(base_valset, batch_size=args.bsize * 2,
                                                 num_workers=args.num_workers,
                                                 shuffle=False, drop_last=False)
    ############################


    performance_val = validate(pretrained_model, clf,
                               base_valloader, vallog, device)

    vallog.save()


def load_checkpoint(model, clf, optimizer, scheduler, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    scheduler.load_state_dict(sd['scheduler'])

    return sd['epoch']


def validate(model, clf,
             base_loader, testlog, device, postfix='Validation'):
    meters = utils.AverageMeterSet()
    model.to(device)
    model.eval()
    clf.eval()

    loss_ce = nn.CrossEntropyLoss()
    end = time.time()
    logits_base_all = []
    ys_base_all = []
    with torch.no_grad():
        # Compute the loss on the source base dataset
        for X_base, y_base in base_loader:            
            X_base = X_base.to(device)
            y_base = y_base.to(device)

            features = model(X_base)
            logits_base = clf(features)
            
            logits_base_all.append(logits_base)
            ys_base_all.append(y_base)

    ys_base_all = torch.cat(ys_base_all, dim=0)
    logits_base_all = torch.cat(logits_base_all, dim=0)

    loss_base = loss_ce(logits_base_all, ys_base_all)
    loss = loss_base 

    meters.update('CE_Loss_source_test', loss_base.item(), 1)

    perf_base = utils.accuracy(logits_base_all.data,
                               ys_base_all.data, topk=(1, ))

    meters.update('top1_base_test', perf_base['average'][0].item(), 1)
    meters.update('top1_base_test_per_class',
                  perf_base['per_class_average'][0].item(), 1)

    meters.update('Batch_time', time.time() - end)

    logger_string = ('{postfix} Batch Time: {meters[Batch_time]:.4f} '
                     'Average CE Loss (Source): {meters[CE_Loss_source_test]: .4f} '
                     'Top1_base_test: {meters[top1_base_test]:.4f} '
                     'Top1_base_test_per_class: {meters[top1_base_test_per_class]:.4f} ').format(
        postfix=postfix, meters=meters)

    testlog.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    testlog.record(0, {
        **values,
        **averages,
        **sums
    })

    if postfix != '':
        postfix = '_' + postfix

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STARTUP')
    parser.add_argument('--dir', type=str, default='./logs/vanilla/EuroSAT',
                        help='directory to save the checkpoints')

    parser.add_argument('--bsize', type=int, default=32,
                        help='batch_size for STARTUP')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='Frequency (in epoch) to evaluate on the val set')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--embedding_load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')    
    parser.add_argument('--embedding_load_path_version', type=int, default=1,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    parser.add_argument('--iteration_bp', type=int,
                        help='which step to break in the training loop')
    parser.add_argument('--model', type=str, default='resnet10',
                        help='Backbone model')

    parser.add_argument('--backbone_random_init', action='store_true',
                        help="Use random initialized backbone ")

    parser.add_argument('--base_dataset', type=str,
                        default='ImageNet_test', help='base_dataset to use')
    parser.add_argument('--base_path', type=str,
                        default=ImageNet_test_path, help='path to base dataset')
    parser.add_argument('--base_split', type=str,
                        help='split for the base dataset')
    parser.add_argument('--base_no_color_jitter', action='store_true',
                        help='remove color jitter for ImageNet')
    parser.add_argument('--base_val_ratio', type=float, default=0.05,
                        help='amount of base dataset set aside for validation')

    parser.add_argument('--batch_validate', action='store_true',
                        help='to do batch validate rather than validate on the full dataset (Ideally, for SimCLR,' +
                        ' the validation should be on the full dataset but might not be feasible due to hardware constraints')

    parser.add_argument('--target_dataset', type=str, default='EuroSAT',
                        help='the target domain dataset')
    parser.add_argument('--target_subset_split', type=str, 
                        help='path to the csv files that specifies the unlabeled split for the target dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')

    args = parser.parse_args()
    main(args)
