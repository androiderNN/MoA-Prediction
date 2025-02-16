import torch
import torch.nn as nn

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
        return -1 * torch.mean((torch.log(pred) * target) + (torch.log(1 - pred) * (1 - target)))

def get_lossfn(fn_name):
    '''
    損失関数を返す'''
    lossfn = None

    if fn_name == 'logloss':
        lossfn = log_loss()
    if fn_name == 'crossentropy':
        lossfn = nn.CrossEntropyLoss()
    elif fn_name == 'mse':
        lossfn = nn.MSELoss()
    else:
        raise ValueError(f'loss function {fn_name} not defined.')

    return lossfn

def get_optimizer(opt_name):
    '''
    optimizerを返す'''
    opt = None

    if opt_name == 'adam':
        opt = torch.optim.Adam()
    else:
        raise ValueError(f'optimizer {opt_name} not defined.')

    return opt

class model_torch(model_base):
    '''
    torchを使用するモデルが継承するクラス
    学習ループ等を実装済み'''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        arg_keys = ['num_iter', 'loss_fn', 'optimizer', 'batch_size']
        self.checkarguments(arg_keys)

        self.model = None
        self.loss_fn = get_lossfn(kwargs['loss_fn'])
        self.optimizer = get_optimizer(kwargs['optimizer'])

    def train(self, x, y):
        '''
        model等を設定すれば学習ループを回す'''
        self.model.train()

        for i in range(self.kwargs['num_iter']):
            self.optimizer.zero_grad()

            # 予測
            y_pred = self.model(x)

            # 逆伝播
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

            if i % (self.kwargs['num_iter'] // 10) == 0:
                print(f'{i}th iter')

    def predict(self, x):
        self.model.eval()
        return self.model(x)

