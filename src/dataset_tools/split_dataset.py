import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_validation(sig_ids : list[str], test_size : float = 0.2, random_state : int = 0):
    '''
    sig_idのリストを渡すと分割したリストを返す'''
    tr_ids, va_ids = train_test_split(sig_ids, test_size=test_size, random_state=random_state)
    return tr_ids, va_ids
