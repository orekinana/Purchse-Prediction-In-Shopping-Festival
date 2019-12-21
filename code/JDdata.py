import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import datetime


class JD(Dataset):
    def __init__(self, seqlen, area, time):
        self.seqlen = seqlen

        self.historical = load_data_dynamic('purchase', seqlen, area, time)
        self.support = load_data_dynamic('cart', seqlen, area, time)

        # self.region = load_data_region('region', seqlen, area, time)

        self.target = load_data_target('purchase', seqlen, area, time)  


        print('Historical data:', self.historical.shape, 'Support data:', self.support.shape, 'Target data:', self.target.shape)

    def __getitem__(self, index):
        historical = self.historical[:,index]
        support = self.support[index]

        # region = self.region[index]

        target = self.target[index]

        return torch.FloatTensor(historical), torch.FloatTensor(support), torch.FloatTensor(target)

    def __len__(self):
        return len(self.target)

def getEveryDay(begin_date,end_date):
    date_list = []
    date_flag = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        if begin_date.strftime("%m-%d") >= '11-01' and begin_date.strftime("%m-%d") <= '11-11':
            date_flag.append(11)
        elif begin_date.strftime("%m-%d") >= '12-01' and begin_date.strftime("%m-%d") <= '12-12':
            date_flag.append(12)
        elif begin_date.strftime("%m-%d") >= '10-01' and begin_date.strftime("%m-%d") <= '10-07':
            date_flag.append(10)
        elif begin_date.strftime("%w") == '0' or begin_date.strftime("%w") == '6':
            date_flag.append(1)
        else:
            date_flag.append(0)
        begin_date += datetime.timedelta(days=1)
    return date_list, date_flag

def load_data_dynamic(data_type, seq_len, area, time):
    # data shape: batch * seqlen * feature
    begin_date, end_date = '2015-01-01', '2019-12-18'
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)

    if area == 'all':
        regions = os.listdir(datadir)
        regions.sort()
        date_flag = np.array(date_flag)
        time_indexs = np.where(date_flag == time)[0]
        output = []
        for region in regions:
            data = np.load(datadir + '/' + region) 
            for time_index in time_indexs:
                if time_index >= seq_len:
                    output.append(data[time_index-seq_len:time_index])
                elif time_index > 0 and time_index < seq_len:
                    pad_num = seq_len-time_index
                    temp_data = data[:time_index]
                    output.append(np.pad(temp_data,((pad_num,0),(0,0)),'edge'))
                else:
                    continue
        
    if time == 'all': # deal all time in one area
        data = np.load(datadir + '/' + area + '.npy')
        output = []
        for i in range(seq_len, len(date_list)):
            output.append(data[i-seq_len:i])

    output = np.array(output)
    # scaler = MinMaxScaler().fit(output)
    # output = scaler.transform(output)
    return output

def load_data_target(data_type, seq_len, area, time):
    begin_date, end_date = '2015-01-01', '2019-12-18'
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)

    if area == 'all':
        regions = os.listdir(datadir)
        regions.sort()
        date_flag = np.array(date_flag)
        time_indexs = np.where(date_flag == time)[0]
        output = []
        for region in regions:
            data = np.load(datadir + '/' + region)
            output.extend(data[time_indexs])

    if time == 'all':
        data = np.load(datadir + '/' + area + '.npy')
        output = data[seq_len:len(date_list)]

    output = np.array(output)
    # scaler = MinMaxScaler().fit(output)
    # output = scaler.transform(output)
    return output

def load_data_region(data_type, seq_len, area, time):
    begin_date, end_date = '2015-01-01', '2019-12-18'
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)
    data = pd.read_csv(datadir + data_type + '.csv')
    if area == 'all':
        regions = os.listdir(datadir)
        regions.sort()
        for region in regions:
            pad_num = len(np.where(date_flag == time)[0])
            output = np.pad(data[data['市'] == region.split('.')[0]],((pad_num,0),(0,0)),'edge')
    if time == 'all':
        pad_num = len(date_flag) - seq_len
        output = np.pad(data[data['市'] == area],((pad_num,0),(0,0)),'edge')
    return output


if __name__ == "__main__":
    dataset = JD(seqlen=7, area='all', time=12)