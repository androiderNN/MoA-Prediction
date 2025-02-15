from pathlib import Path
import pandas as pd

import config
from dataset_tools import tabular

def save_tabular_data():
    '''
    表データを保存する'''
    tabular.save_x(config.train_paths['raw_features'], config.train_paths['tabular_x'])
    tabular.save_x(config.test_paths['raw_features'], config.test_paths['tabular_x'])

def save_y():
    '''
    targetを保存する'''
    df = pd.read_csv(config.train_paths['raw_target_scored']).drop(columns='sig_id')
    df.to_csv(config.train_paths['y'])

def save_ids():
    '''
    id列のみ保存する'''
    for paths in [config.train_paths, config.test_paths]:
        ids = pd.read_csv(paths['raw_features'], usecols=['sig_id'])
        ids.to_csv(paths['sig_id'])

def main():
    save_y()
    save_ids()
    save_tabular_data()

if __name__ == '__main__':
    main()