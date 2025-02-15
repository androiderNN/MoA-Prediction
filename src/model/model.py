class model_base():
    def __init__(self):
        self.model = None
        self.kwargs :dict = None

    def argumentcheck(self, arg_keys):
        for k in arg_keys:
            if k not in self.kwargs.keys():
                raise ValueError(f'kwargs need {k}')
    
    def train(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError