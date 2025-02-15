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