import torch.nn as nn

from . import model

class linear_3layers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=kwargs['num_features'], out_features=kwargs['hidden_size']),
            nn.Dropout(p=kwargs['p_dropout']),
            nn.Linear(in_features=kwargs['hidden_size'], out_features=kwargs['out_features'])
        )

    def forward(self, x):
        return self.net(x)

class model_tabular_nn(model.model_torch):
    '''
    全結合層による予測モデル 主に表データを想定'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.checkarguments(['linear_3layers_params'])

        self.model = linear_3layers(**kwargs['linear_3layers_params'])
        self.set_optimizer()
