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

def find_region(box, lng, lat):
    for key in box:
        if lat > box[key]['s'] and lat < box[key]['n'] and lng > box[key]['w'] and lng < box[key]['e']:
            return key

def write_item(historical, item_list):

    with open('../data/geohash6_orderlist.txt', 'w') as f:
        json.dump(item_list, f)
    
    for region in historical:
        if not os.path.exists('../data/' + str(region)):
            os.makedirs('../data/' + str(region))
        item_strs = []
        for date in historical[region]:
            item_str = []
            for item in item_list:
                if item in historical[region][date]:
                    item_str.append(historical[region][date][item])
                else:
                    item_str.append(0)
            item_strs.append(item_str)
        np.save('../data/' + str(region) + '/geohash6_order.npy', np.array(item_strs))



if __name__ == "__main__":

    region_num = 60

    with open('../data/gid2bbox.txt') as f:
        line = f.readline()
        lines = f.readlines()
    box = {}
    for line in lines:
        current_region = line.strip().split(',')
        number = int(current_region[0])
        box[current_region[0]] = {'s':current_region[1], 'w':current_region[2], 'n':current_region[3], 'e':current_region[4]}

    # dates = getEveryDay('20160801', '20190717')
    dates = getEveryDay('20180801', '20181031')
    date_dict = {}
    for date in dates:
        date_dict[date] = {}
    historical = {str(i):copy.deepcopy(date_dict) for i in range(region_num)}

    with open('../data/data.csv','r') as f:
        lines = f.readlines()
    item_list1 = {}

    for line in lines:
        date, lng, lat, product, c1, c2, c3  = line.strip().split('\t')[1:]
        item_list1[c1] = 1
        year, month, day = date.split(' ')[0].split('-')
        date = year+month+day
        region = find_region(box, lng, lat)
        # print('region:', region,'date:', date,'item:', c1)
        if region not in historical:
            continue
        if date not in historical[region]:
            continue
        if c1 not in historical[region][date]:
            historical[region][date][c1] = 0
        historical[region][date][c1] += 1

    item_list1 = [k for k, v in item_list1.items()] 
    write_item(historical, item_list1)
