import yaml
import datetime
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

import config
from model import gbdt, tabular_nn
from dataset_tools import dataset, split_dataset, tabular

now = datetime.datetime.now()

def load_params() -> dict:
    '''
    yamlファイル記載のparametersを読み込む'''
    with open(config.param_yaml) as f:
        params = yaml.safe_load(f)
    return params

def export_result(pred : np.array) -> None:
    '''
    結果を保存する
    予測値、params.yaml'''
    # path設定
    dir = datetime.datetime.strftime(now, '%m-%d-%H-%M-%S')
    expath = config.export_dir / dir
    expath.mkdir()

    # データフレーム保存
    df = pd.read_csv(config.test_paths['sample_submission'], nrows=0)
    df['sig_id'] = dataset.get_id_list(config.test_paths['sig_id'])
    df.iloc[:, 1:] = pred
    df = df.astype({c: object if c == 'sig_id' else np.float16 for c in df.columns})    # 型変換
    df.to_csv(expath / 'submission.csv', index=False, float_format='%.5f')  # 桁数を指定して保存

    # params.yamlをコピー
    shutil.copy(config.param_yaml, expath / 'params.yaml')

    print(f'file saved : {expath}')

def train_tabular_nn():
    params = load_params()

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
    model = tabular_nn.model_tabular_nn(**params['model_tabular_nn'])
    model.train(train_dataset, valid_dataset)
    print('\ntraining finished.\n')

    # 予測
    test_pred = model.predict(test_dataset)
    export_result(test_pred)

if __name__ == '__main__':
    train_tabular_nn()
