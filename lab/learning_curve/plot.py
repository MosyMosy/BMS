
import os
import glob
from numpy import tile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

methods = ['STARTUP_na'] #['vanilla', 'BMS_in', 'BAS_in', 'baseline', 'baseline_na']
target_datasets = ['EuroSAT', 'CropDisease', 'ISIC', 'ChestX']

def get_logs_path(method, target):
    root = './logs'
    method_path = root + '/' + method
    if os.path.isdir(method_path) == False:
        print('The methode {}\'s  path doesn\'t exist'.format(method))        
    log_path = method_path + '/' + target
    if os.path.isdir(log_path) == False:
        log_path = method_path
    return log_path


def plot_all():
    for method in methods:
        for target in target_datasets:
            log_path = get_logs_path(method, target)
            best_check = log_path + '/' + 'checkpoint_best.pkl'
            train_log = glob.glob(log_path + '/' + 'train_*.csv')
            val_log = glob.glob(log_path + '/' + 'val_*.csv')
            if (len(train_log) == 0) or (len(val_log) == 0):
                raise ValueError('The path {} does not contain logs'.format(log_path))
                continue
            elif (len(train_log) > 1) or (len(val_log) > 1):
                raise ValueError('The path {} contains extra logs'.format(log_path))
                continue
            else:
                train_log = train_log[0].replace('\\', '/')
                val_log = val_log[0].replace('\\', '/')
                df = pd.read_csv(val_log)
                columns = df.columns
                df = pd.DataFrame(np.repeat(df.values,2,axis=0))
                df.columns = columns
                df['Loss_train'] = pd.read_csv(train_log)['Loss']
                df.plot( y=["Loss_train", 'Loss_test'])
                df.plot( y=["top1_base_test"] )
                plt.title('{0}_{1}'.format(method, target))
                plt.show()

def compare_baselines():
    ['baseline', 'baseline_na']
    baseline_method = 'baseline'
    baseline_na_method = 'baseline_na'
    root = './logs'
    baseline_path = root + '/' + baseline_method
    baseline_na_path = root + '/' + baseline_na_method
    
    baseline_check = baseline_path + '/' + 'checkpoint_best.pkl'
    baseline_na_check = baseline_na_path + '/' + 'checkpoint_best.pkl'
    baseline_train_log = glob.glob(baseline_path + '/' + 'train_*.csv')
    baseline_val_log = glob.glob(baseline_path + '/' + 'val_*.csv')
    baseline_na_train_log = glob.glob(baseline_na_path + '/' + 'train_*.csv')
    baseline_na_val_log = glob.glob(baseline_na_path + '/' + 'val_*.csv')
    
    baseline_train_log = baseline_train_log[0].replace('\\', '/')
    baseline_val_log = baseline_val_log[0].replace('\\', '/')
    baseline_na_train_log = baseline_na_train_log[0].replace('\\', '/')
    baseline_na_val_log = baseline_na_val_log[0].replace('\\', '/')
    
    df_baseline_train = pd.read_csv(baseline_train_log)
    df_baseline_val = pd.read_csv(baseline_val_log)
    df_baseline_na_train = pd.read_csv(baseline_na_train_log)
    df_baseline_na_val = pd.read_csv(baseline_na_val_log)
    
    columns = df_baseline_val.columns
    df_baseline_val = pd.DataFrame(np.repeat(df_baseline_val.values,2,axis=0))
    df_baseline_na_val = pd.DataFrame(np.repeat(df_baseline_na_val.values,2,axis=0))   
    df_baseline_val.columns = columns 
    df_baseline_na_val.columns = columns
        
    df = pd.DataFrame()
    df['baseline_train_loss'] = df_baseline_train['Loss']
    df['baseline_na_train_loss'] = df_baseline_na_train['Loss']
    df.plot( y=["baseline_train_loss", 'baseline_na_train_loss'])
    plt.axvline(x=354, color='blue')
    plt.axvline(x=332, color='orange')
    
    df['baseline_val_loss'] = df_baseline_val['Loss_test']
    df['baseline_na_val_loss'] = df_baseline_na_val['Loss_test']
    df.plot( y=["baseline_val_loss", 'baseline_na_val_loss'])
    plt.axvline(x=354, color='blue')
    plt.axvline(x=332, color='orange')
    
    df['baseline_val_top1'] = df_baseline_val['top1_base_test']
    df['baseline_na_val_top1'] = df_baseline_na_val['top1_base_test']
    df.plot( y=["baseline_val_top1", 'baseline_na_val_top1'])
    plt.axvline(x=354, color='blue')
    plt.axvline(x=332, color='orange')
    
    plt.show()
    
plot_all()
# compare_baselines()
