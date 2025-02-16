import yaml
import pandas as pd

import config
from model import gbdt, tabular_nn
from dataset_tools import dataset, split_dataset, tabular

def load_params():
    '''
    yamlファイル記載のparametersを読み込む'''
    with open(config.param_yaml) as f:
        params = yaml.safe_load(f)
    return params

def train_tabular_nn():
    # validation分割
    train_sig_ids = dataset.get_id_list(config.train_paths['sig_id'])
    tr_ids, va_ids = split_dataset.split_validation(train_sig_ids)

    # ロード
    train_dataset = tabular.tabular_dataset(
        config.train_paths['tabular_x'],
        config.train_paths['sig_id'],
        y_path=config.train_paths['y'],
        sig_id=tr_ids,
    )
    valid_dataset = tabular.tabular_dataset(
        config.train_paths['tabular_x'],
        config.train_paths['sig_id'],
        y_path = config.train_paths['y'],
        sig_id=va_ids,
    )

    test_dataset = tabular.tabular_dataset(
        config.test_paths['tabular_x'],
        config.test_paths['sig_id'],
    )

    # 学習
    params = load_params()
    model = tabular_nn.model_tabular_nn(**params['model_tabular_nn'])
    model.train(train_dataset, valid_dataset)

if __name__ == '__main__':
    train_tabular_nn()
