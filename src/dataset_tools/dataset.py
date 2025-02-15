
class dataset():
    '''
    各形式のデータセットが継承するクラス'''
    def __init__(self):
        self.x = None
        self.y = None
        self.sig_id = None
    
    def get_id_list(self) -> list[str]:
        '''sig_idのリストを取得する'''
        return self.sig_id

    def id_to_index(self, id : str) -> int:
        '''
        idからindexを取得する'''
        raise NotImplementedError
    
    def get_x(self):
        '''
        indexで学習データを取得するメソッド'''
        raise NotImplementedError
    
    def get_x_byid(self, id : list[str]):
        '''
        idで学習データを取得するメソッド'''
        return [self.get_x(self.id_to_index(i)) for i in id]
    
    def get_y(self):
        '''
        indexでターゲットを取得するメソッド'''
        raise NotImplementedError

    def get_y_byid(self, id : list[str]):
        '''
        idでターゲットを取得するメソッド'''
        return [self.get_y(self.id_to_index(i)) for i in id]
    
