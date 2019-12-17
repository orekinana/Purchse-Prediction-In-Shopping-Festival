import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import torch


class JD(Dataset):
    def __init__(self, seqlen):
        self.seqlen = seqlen

        self.historical = load_data('geohash6_order', seqlen)
        # self.support = load_data('support', seqlen)

        self.region = load_data('region', seqlen)
        # self.POIs = load_data('POI', seqlen)

        self.target = load_data('target', seqlen)  
        print(self.target)                                                

# print('Historical data:', self.historical.shape, 'Regoin data:', self.region.shape, 'Target data:', self.target.shape)

    def __getitem__(self, index):
        historical = self.historical[:,index]
        # support = self.support[index]

        region = self.region[index]
        # poi = self.POIs[index]

        target = self.target[index]

        return torch.FloatTensor(historical), torch.FloatTensor(region), torch.FloatTensor(target)

    def __len__(self):
        return len(self.target)

def load_data(feature_name, seq_len):
    # batch * seqlen * feature
    if feature_name == 'geohash6_order':
        # print(feature_name)
        data = []
        for grid in range(60):
            if grid == 54:
                continue
            order = np.load('../data/0/' + feature_name + '.npy')
            # normalize data
            scaler = MinMaxScaler().fit(order)
            order = scaler.transform(order)

            temp = []
            for i in range(order.shape[0]-seq_len):
                temp.append(order[i:i+seq_len])
            data.extend(np.array(temp))
        data = np.array(data).transpose(1,0,2)
        return data

    if feature_name == 'region' or 'target' or 'POI':
        # print(feature_name)
        if feature_name == 'target':
            feature_name = 'geohash6_order'
        data = np.load('../data/0/' + feature_name + '.npy')[seq_len:]
        for grid in range(1, 60):
            if grid == 54:
                continue
            data = np.concatenate((data, np.load('../data/' + str(grid) + '/' + feature_name + '.npy')[seq_len:]), axis=0)
        # normalize data
        scaler = MinMaxScaler().fit(data)
        data = scaler.transform(data)
        print(feature_name, np.max(data), np.min(data), np.median(data))
        return data

if __name__ == "__main__":
    dataset = JD(seqlen=10)