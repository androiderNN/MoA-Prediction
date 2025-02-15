import numpy as np
import pandas as pd

from . import dataset

class tabular_dataset(dataset.dataset):
    '''
    表形式データのデータセットクラス'''
    def __init__(self, x_path, sig_id, y_path = None):
        self.x = pd.read_csv(x_path).to_numpy(dtype='float16')
        sig_id = pd.read_csv(sig_id).to_numpy(dtype='object')
        self.id2ind_dic = {id: i for id, i in enumerate(sig_id)}

        # trainデータのみ
        self.y = pd.read_csv(y_path).to_numpy(dtype='float16') if y_path is not None else None

    def id_to_index(self, ids : list[str]) -> list[int]:
        return [self.id2ind_dic[id] for id in ids]

    def get_x(self, index : list[int]):
        return self.x[index]

    def get_y(self, index : list[int]):
        return self.y[index]

def save_x(csvpath, savepath) -> None:
    '''
    csvpathのcsvを読み、クラス変数をバイナリエンコーディングしてsavepathに保存する'''
    df = pd.read_csv(csvpath)

    # cp_type : 'trt_cp' or 'ctl_vehicle'
    df['cp_type'] = df['cp_type'] == 'trt_cp'

    # cp_time : 24 or 48 or 72
    # df['cp_time'] = df['cp_time'] / 24

    # cp_dose : D1 or D2
    df['cp_dose'] = df['cp_dose'] == 'D1'

    # id列削除
    df.drop(columns='sig_id', inplace=True)

    # 保存
    df.to_csv(savepath)
