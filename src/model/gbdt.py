import lightgbm as lgb

from . import model

class model_lgb(model.model_base):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def train(self, tr_x, tr_y, va_x, va_y):
        tr_lgb = lgb.Dataset(tr_x, tr_y)
        va_lgb = lgb.Dataset(va_x, va_y)

        self.model = lgb.train(
            params=self.kwargs['params'],
            train_set=tr_lgb,
            valid_sets=[va_lgb],
            valid_names=['train', 'valid'],
            verbose=False,
            callbacks=[lgb.early_stopping(stopping_rounds=self.kwargs['stopping_rounds'], vebose=True)]
        )

    def predict(self, x):
        return self.model.predict(x)
