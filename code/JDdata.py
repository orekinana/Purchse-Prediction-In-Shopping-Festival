import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import datetime
import json

# total dates
traning_op, training_ed = '2015-01-01', '2018-12-31'

# training date
testing_op, testing_ed = '2019-01-01', '2019-12-17'

# total regions
regions = ['怀柔区', '昌平区', '大兴区', '朝阳区', '延庆区', '宣武区', '石景山区', '平谷区', '西城区', '东城区', '顺义区', '通州区', '丰台区']

class JD(Dataset):
    def __init__(self, seqlen, area, time, mode):
        if mode == 'training':
            date_op, date_ed = traning_op, training_ed
        else:
            date_op, date_ed = testing_op, testing_ed
        self.seqlen = seqlen

        self.historical1 = load_data_dynamic('purchase', seqlen, area, time, 1, date_op, date_ed, mode)
        self.historical2 = load_data_dynamic('purchase', seqlen, area, time, 7, date_op, date_ed, mode)
        self.historical3 = load_data_dynamic('purchase', seqlen, area, time, 30, date_op, date_ed, mode)

        self.support1 = load_data_dynamic('cart', seqlen, area, time, 1, date_op, date_ed, mode)
        self.support2 = load_data_dynamic('cart', seqlen, area, time, 7, date_op, date_ed, mode)
        self.support3 = load_data_dynamic('cart', seqlen, area, time, 30, date_op, date_ed, mode)

        self.static = load_data_static('static', seqlen, area, time, date_op, date_ed, mode)

        self.target = load_data_target('purchase', seqlen, area, time, date_op, date_ed, mode)  


        # print('Historical1 data:', self.historical1.shape, 'Historical2 data:', self.historical2.shape, 'Historical3 data:', self.historical3.shape, \
        #         'Support1 data:', self.support1.shape, 'Support2 data:', self.support2.shape, 'Support3 data:', self.support3.shape, \
        #             'Target data:', self.target.shape, 'Static data:', self.static.shape)

    def __getitem__(self, index):
        historical1 = self.historical1[index]
        historical2 = self.historical2[index]
        historical3 = self.historical3[index]

        support1 = self.support1[index]
        support2 = self.support2[index]
        support3 = self.support3[index]

        static = self.static[index]

        target = self.target[index]

        return torch.FloatTensor(historical1), torch.FloatTensor(historical2), torch.FloatTensor(historical3), \
                torch.FloatTensor(support1), torch.FloatTensor(support2), torch.FloatTensor(support3), \
                    torch.FloatTensor(static), torch.FloatTensor(target)

    def __len__(self):
        return len(self.target)

def getEveryDay(begin_date,end_date):
    date_list = []
    date_flag = []
    skip_day = begin_date
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        if begin_date.strftime("%Y-%m-%d") == skip_day:
            date_flag.append(-1)
        elif begin_date.strftime("%m-%d") >= '11-01' and begin_date.strftime("%m-%d") <= '11-11':
            date_flag.append(11)
        elif begin_date.strftime("%m-%d") >= '12-01' and begin_date.strftime("%m-%d") <= '12-12':
            date_flag.append(12)
        elif begin_date.strftime("%m-%d") >= '10-01' and begin_date.strftime("%m-%d") <= '10-07':
            date_flag.append(10)
        elif begin_date.strftime("%m-%d") >= '06-01' and begin_date.strftime("%m-%d") <= '06-20':
            date_flag.append(618)
        elif begin_date.strftime("%w") == '0' or begin_date.strftime("%w") == '6':
            date_flag.append(1)
        else:
            date_flag.append(0)
        begin_date += datetime.timedelta(days=1)
    return date_list, date_flag

def load_data_dynamic(data_type, seq_len, area, time, skip_day, begin_date, end_date, mode):
    # data shape: batch * seqlen * feature
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)
    output = []
    if area == 'all':
        date_flag = np.array(date_flag)
        time_indexs = np.where(date_flag == time)[0]
        for region in regions:
            data = np.load(datadir + '/' + mode + '_' + region + '.npy') 
            for time_index in time_indexs:
                has_num = int(time_index / skip_day) + 1
                if has_num >= seq_len:
                    has_num = seq_len
                pad_num = seq_len - has_num
                temp = []
                for i in range(has_num):
                    temp.append(data[time_index-i*skip_day-1])
                output.append(np.pad(np.array(temp),((pad_num,0),(0,0)),'edge'))
        
    if time == 'all': # deal all time in one area
        data = np.load(datadir + '/' + mode + '_' + area + '.npy')
        for time_index in range(1, len(date_list)):
            has_num = int(time_index / skip_day) + 1
            if has_num >= seq_len:
                has_num = seq_len
            pad_num = seq_len - has_num
            temp = []
            for i in range(has_num):
                temp.append(data[time_index-i*skip_day-1])
            output.append(np.pad(np.array(temp),((pad_num,0),(0,0)),'edge'))

    output = np.array(output)
    
    return output

def load_data_target(data_type, seq_len, area, time, begin_date, end_date, mode):
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)
    output = []
    if area == 'all':
        date_flag = np.array(date_flag)
        time_indexs = np.where(date_flag == time)[0]
        
        for region in regions:
            data = np.load(datadir + '/' + mode + '_' + region + '.npy')
            output.extend(data[time_indexs])

    if time == 'all':
        data = np.load(datadir + '/' + mode + '_' + area + '.npy')
        output = data[1:len(date_list)]

    output = np.array(output)
    return output

def load_data_static(data_type, seq_len, area, time, begin_date, end_date, mode):
    datadir = '../data/spatial/' + data_type
    date_list, date_flag = getEveryDay(begin_date, end_date)

    # select by region
    with open(datadir + '/new_poi.json', 'r') as f:
        poi = json.load(f)
    # select by year+region
    with open(datadir + '/user_portrait.json', 'r') as f:
        user = json.load(f)
    festival = np.load(datadir + '/' + mode + '_festival_feature.npy') # select by date (same with date_flag index)
    
    output = []
    if area == 'all':
        date_flag = np.array(date_flag)
        time_indexs = np.where(date_flag == time)[0]
        for region in regions:
            for time_index in time_indexs:
                year = 2015 + int(time_index / 365)
                output.append(poi[region] + user[str(year) + region] + list(festival[time_index]))
    if time == 'all':
        for i in range(1, len(date_list)):
            year = 2015 + int(i / 365)
            output.append(poi[area] + user[str(year) + area] + list(festival[i]))
    
    output = np.array(output)
    output = MinMaxScaler().fit_transform(output)
    return output


if __name__ == "__main__":
    dataset = JD(seqlen=7, area='all', time=12, mode='training')