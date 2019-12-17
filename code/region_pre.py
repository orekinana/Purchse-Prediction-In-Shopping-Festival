import numpy as np
import pandas as pd
import json
import datetime
import os
import copy

def getEveryDay(begin_date,end_date):

    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list

if __name__ == "__main__":
    region_num = 60

    region_age = pd.read_csv('../data/geohash6_age_pre.csv.txt')
    region_gender = pd.read_csv('../data/geohash6_gender_pre.csv.txt')
    region_purchpower = pd.read_csv('../data/geohash6_purchpower_pre.csv.txt')

    region_age['time'] = region_age['time'].map(lambda x: x//100)
    region_gender['time'] = region_gender['time'].map(lambda x: x//100)
    region_purchpower['time'] = region_purchpower['time'].map(lambda x: x//100)

    region_age = region_age.groupby(['time', 'gid'])['age_15','age_16_25','age_26_35','age_36_45','age_46_55','age_56'].sum().reset_index()
    region_gender = region_gender.groupby(['time', 'gid'])['man','woman'].sum().reset_index()
    region_purchpower = region_purchpower.groupby(['time', 'gid'])['土豪','高级白领','小白领','蓝领','收入很少'].sum().reset_index()

    for i in range(region_num):

        region_age_np = np.array(region_age[region_age['gid']==i])[:, 2:].astype("float")
        region_gender_np = np.array(region_gender[region_gender['gid']==i])[:, 2:].astype("float")
        region_purchpower_np = np.array(region_purchpower[region_purchpower['gid']==i])[:, 2:].astype("float")
        
        region_data = np.concatenate([region_age_np, region_gender_np, region_purchpower_np], axis=1)
        np.save('../data/' + str(i) + '/region.npy', region_data)
        print('region ' + str(i) + ' data shape:', region_data.shape)