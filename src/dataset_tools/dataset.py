
class dataset():
    '''
    各形式のデータセットが継承するクラス'''
    def __init__(self):
        self.x = None
        self.y = None
    
    def id_to_index(self, id : str) -> int:
        '''
        idからindexを取得する'''
        raise NotImplementedError
    
    def get_id(self, index : list[int]) -> list[str]:
        '''
        indexからidを取得する'''
        return [self.id_to_index[i] for i in index]

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
    
