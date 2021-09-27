
import os


methods = ['vanilla', 'BMS', 'BMS_in', 'BAS_in', 'baseline', 'baseline_na']
target_datasets = ['EuroSAT', 'CropDisease', 'ISIC', 'ChestX']

def get_logs_path(method, target):
    root = './logs'
    method_path = root + '/' + method
    if os.path.isdir(method_path) == False:
        raise ValueError('The methode {}\'s  path doesn\'t exist'.format(method))
        return
    log_path = method_path + '/' + target
    if os.path.isdir(log_path) == False:
        log_path = method_path
    return log_path

for method in methods:
    for target in target_datasets:
        log_path = get_logs_path(method, target)
        
