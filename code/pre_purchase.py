import numpy as np
import pandas as pd
import datetime

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

def spatial_partition(purchase_data, cart_data, cate_list, area_flag, date_list):
    for area in area_flag:
        area_cart = []
        for date in date_list:
            date_cart = []
            for cate in cate_list:
                entry = cart_data[(cart_data['日期'] == date) & (cart_data['市'] == area) & (cart_data['一级类目'] == cate)]['数量']
                if len(entry) == 0:
                    date_cart.append(0)
                else:
                    date_cart.append(int(entry))
            area_cart.append(date_cart)
            print(area, date)
        np.save('../data/spatial/cart/' + area + '.npy', np.array(area_cart))

if __name__ == "__main__":

    data_dir = '../data/'
    begin_date, end_date = '2015-01-01', '2019-12-18'
    begin_year, end_year = 2015, 2020
    dates, date_flag = getEveryDay(begin_date, end_date)

    areas = ['怀柔区', '大兴区', '昌平区', '通州区', '石景山区', '延庆区', '顺义区', '西城区', '平谷区', '宣武区', '东城区', '丰台区', '朝阳区', '门头沟', '崇文区', '房山区', '密云区', '海淀区']
    categories = ['厨具', '服饰内衣', '家用电器', '鞋靴', '家装建材', '宠物生活', '电脑、办公', '图书', '家具', '手机通讯', '玩具乐器', '本地生活/旅游出行', '钟表', '酒类', '影视', '医药保健', '农资园艺', '美妆护肤', '整车', '文娱', '礼品', '珠宝首饰', '食品饮料', '家居日用', '汽车用品', '运动户外', '生鲜', '音乐', '母婴', '数码']
    for year in range(begin_year, end_year):
        purchase_data = pd.read_csv(data_dir + 'purchase-' + str(year) + '.csv', sep = '\t', error_bad_lines=False)
        purchase_data = purchase_data.drop(['省', '区'], axis=1)
        purchase_data['一级类目'] = purchase_data['一级类目'].str.strip()
        purchase_data = purchase_data.groupby(['日期', '市', '一级类目']).sum().reset_index()
        
        cart_data = pd.read_csv(data_dir + 'shoppingcart-' + str(year) + '.csv', sep = '\t', error_bad_lines=False)
        cart_data = cart_data.drop(['省', '区'], axis=1)
        cart_data['一级类目'] = cart_data['一级类目'].str.strip()
        cart_data = cart_data.groupby(['日期', '市', '一级类目']).sum().reset_index() 
        print(len(purchase_data), len(cart_data))
        
        if year == begin_year:
            purchase = purchase_data
            cart = cart_data
        else:
            purchase = pd.concat([purchase, purchase_data])
            cart = pd.concat([cart, cart_data])
    print(len(purchase), len(cart))
    spatial_partition(purchase_data, cart_data, categories, areas, dates)
