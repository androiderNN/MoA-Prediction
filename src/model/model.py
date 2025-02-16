import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class model_base():
    '''
    各モデルを実行するクラスのベース'''
    def __init__(self):
        self.model = None
        self.kwargs :dict = None

    def checkarguments(self, arg_keys : list) -> None:
        '''
        __init__で渡されたkwargsに必要なkeyが存在するかチャックする
        子クラスの__init__で呼び出す'''
        for k in arg_keys:
            if k not in self.kwargs.keys():
                raise ValueError(f'kwargs need {k}')

    def train(self):
        '''
        x, yを渡すとトレーニングする'''
        raise NotImplementedError

    def predict(self):
        '''
        xを渡すと予測値を返す'''
        raise NotImplementedError

class log_loss(nn.Module):
    '''
    コンペで使用される評価関数'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred + 1e-10
        return -1 * torch.mean((torch.log(pred) * target) + (torch.log(1 - pred) * (1 - target)))

def get_lossfn(fn_name):
    '''
    損失関数を返す'''
    lossfn = None

    if fn_name == 'logloss':
        lossfn = log_loss()
    elif fn_name == 'crossentropy':
        lossfn = nn.CrossEntropyLoss()
    elif fn_name == 'mse':
        lossfn = nn.MSELoss()
    else:
        raise ValueError(f'loss function {fn_name} not defined.')

    return lossfn

def get_optimizer(opt_name):
    '''
    optimizerを返す'''
    optimizer = None

    if opt_name == 'adam':
        optimizer = torch.optim.Adam
    else:
        raise ValueError(f'optimizer {opt_name} not defined.')

    return optimizer

class model_torch(model_base):
    '''
    torchを使用するモデルが継承するクラス
    学習ループ等を実装済み'''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        arg_keys = ['num_iter', 'loss_fn', 'optimizer', 'batch_size', 'lr', 'estop_round']
        self.checkarguments(arg_keys)

        self.model = None
        self.optimizer = None
        self.loss_fn = get_lossfn(kwargs['loss_fn'])

    def set_optimizer(self):
        '''
        model設定後に呼び出すとoptimizerを定義する'''
        if self.model is None:
            raise ValueError('set model')

        opt = get_optimizer(self.kwargs['optimizer'])
        self.optimizer = opt(self.model.parameters(), lr=self.kwargs['lr'])

    def plot_loss(self, history : dict):
        '''
        学習履歴をグラフで描画する
        historyは{'train': [], 'valid': []}を想定'''
        plt.plot(np.array(list(history.values())).T)
        plt.show()

    def train(self, tr_dataset, va_dataset):
        '''
        model等を設定すれば学習ループを回す'''
        # データセットのtensorへの変換　要修正
        tr_dataset.x = torch.tensor(tr_dataset.x, dtype=torch.float)
        tr_dataset.y = torch.tensor(tr_dataset.y, dtype=torch.float)

        va_dataset.x = torch.tensor(va_dataset.x, dtype=torch.float)
        va_dataset.y = torch.tensor(va_dataset.y, dtype=torch.float)

        # 記録用list
        history = {'train': list(), 'valid': list()}

        # trainループ
        for i in range(self.kwargs['num_iter']):
            self.model.train()
            self.optimizer.zero_grad()

            # 予測
            tr_pred = self.model(tr_dataset.get_x())

            # 逆伝播
            loss = self.loss_fn(tr_pred, tr_dataset.get_y())
            loss.backward()
            self.optimizer.step()

            # バリデーションでのloss算出と記録
            self.model.eval()
            with torch.no_grad():
                va_pred = self.model(va_dataset.get_x())
                va_loss = self.loss_fn(va_pred, va_dataset.get_y())

            history['train'].append(loss.item())
            history['valid'].append(va_loss.item())

            # 1またはnum_iter//10iterationごとに進捗出力
            if i % max((self.kwargs['num_iter'] // 10), 1) == 0:
                print(f'{i}th epoch')
                print(f'train loss : {loss.item()}')
                print(f'valid loss : {va_loss.item()}')

            # lossがnanの場合
            if np.isnan(loss.item()):
                print(f'{i}th epoch : loss is nan. break')
                self.plot_loss(history)
                break

            # early stopping
            if i >= self.kwargs['estop_round']:
                if history['valid'][-1] >= history['valid'][-1*self.kwargs['estop_round']]:
                    print(f'early stopping : {i}th epoch')
                    self.plot_loss(history)
                    break

    def predict(self, x):
        self.model.eval()
        return self.model(x)

