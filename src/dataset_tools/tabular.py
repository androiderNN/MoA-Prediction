import numpy as np
import pandas as pd

from . import dataset

class tabular_dataset(dataset.dataset):
    '''
    表形式データのデータセットクラス'''
    def __init__(self, x_path, sig_id_path, y_path = None, sig_id = None):
        self.x = pd.read_csv(x_path).to_numpy(dtype='float16')
        sig_id_master = dataset.get_id_list(sig_id_path)
        self.id2ind_dic = {id: i for i, id in enumerate(sig_id_master)}

        # trainデータのみtargetをロード
        self.y = pd.read_csv(y_path).to_numpy(dtype='float16') if y_path is not None else None

        # validation用に分割した場合を想定
        # idからindexを取得し、必要なデータのみを抽出して再格納
        if sig_id is not None:
            self.x = self.get_x_byid(sig_id)
            self.y = self.get_y_byid(sig_id) if self.y is not None else None

            # sig_idの更新
            self.sig_id = sig_id
            self.id2ind_dic = {id: i for i, id in enumerate(sig_id)}
        else:
            self.sig_id = sig_id_master

    def id_to_index(self, ids : list[str]) -> list[int]:
        return [self.id2ind_dic[id] for id in ids]

    def get_x(self, index : list[int] = None):
        if index is None:
            return self.x
        else:
            return self.x[index]

    def get_y(self, index : list[int] = None):
        if index is None:
            return self.y
        else:
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
    df.to_csv(savepath, index=False)
