import os
import pandas as pd
import numpy as np


PATH = './data/'
TRAINFILE = PATH + 'train.tsv'
TESTFILE = PATH + 'test.tsv'
VALIDFILE = PATH + 'valid.tsv'

def get_item():
    train = pd.read_csv(TRAINFILE, sep='\t', dtype={0:str, 1:str, 2:np.float32})
    valid = pd.read_csv(VALIDFILE, sep='\t', dtype={0:str, 1:str, 2:np.float32})
    test = pd.read_csv(TESTFILE, sep='\t', dtype={0:str, 1:str, 2:np.float32})
    data = pd.concat([train, valid, test])
    return data.ItemId.unique()


def _load_data(f, max_len):
    """
    Data format in file f:
    SessionId\tItemId\tTimestamp\n
    """
    
    if os.path.exists('item2id.map'):
        item2idmap = {}
        for line in open('item2id.map'):
            k, v = line.strip().split('\t')
            item2idmap[k] = int(v)
    else:
        items = get_item()
        item2idmap = dict(zip(items, range(1, 1+items.size))) 
        with open('item2id.map', 'w') as fout:
            for k, v in item2idmap.iteritems():
                fout.write(str(k) + '\t' + str(v) + '\n')
    n_items = len(item2idmap)
    data = pd.read_csv(f, sep='\t', dtype={0:str, 1:str, 2:np.float32})    
    data['ItemId'] = data['ItemId'].map(item2idmap)
    data = data.sort_values(by=['Timestamp']).groupby('SessionId')['ItemId'].apply(list).to_dict()
    new_x = []
    new_y = []
    for k, v in data.items():
        x = v[:-1]
        y = v[1:]
        if len(x) < 2:
            continue
        padded_len = max_len - len(x)
        if padded_len > 0:
            x.extend([0] * padded_len)
            y.extend([0] * padded_len)
        new_x.append(x[:max_len])
        new_y.append(y[:max_len])
    return (new_x, new_y, n_items)

def load_train(max_len):
    return _load_data(TRAINFILE, max_len)

def load_valid(max_len):
    return _load_data(VALIDFILE, max_len)

def load_test(max_len):
    return _load_data(TESTFILE, max_len)

if __name__ == '__main__':
    load_train(20)
