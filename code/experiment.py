import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TemporalRepresentation as TR
from model import SpatioRepresentation as SR
from model import Generation as G
from model import MST
from JDdata import JD
from torch import nn, optim
import argparse
from torch.utils.data import Dataset, DataLoader
import model_configs
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Trainer():
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.initial_lr, weight_decay=0.1)
        print("Trainer initial finish!")

    def train(self, area, time):

        # load data
        dataset = JD(7, area, time, mode='training')
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        data_size = 0
        total_tr_loss = 0
        for epoch in range(self.args.epochs):
            tr_loss = 0
            for batch_idx, (temporal1, temporal2, temporal3, support1, support2, support3, static, target) in enumerate(dataloader):
                data_size = target.shape[1]
                self.model.train()
                pred, spatial_re, temporal_re, mu, sigma = self.model(target, temporal1, temporal2, temporal3, support1, support2, support3, static)
                mse, loss = self.model.loss(target, pred, mu, sigma)
                loss = loss.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # tr_loss += loss.item()
                tr_loss += mse.sum().item()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, tr_loss / (len(dataloader.dataset) * data_size)))
            total_tr_loss += tr_loss / (len(dataloader.dataset) * data_size)
        total_tr_loss /= self.args.epochs
        return total_tr_loss, spatial_re, temporal_re
    
    def generate_temporal(self, area, time):
        # load data
        dataset = JD(7, area, time, mode='training')
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        for epoch in range(self.args.epochs):
            for batch_idx, (temporal1, temporal2, temporal3, support1, support2, support3, static, target) in enumerate(dataloader):
                self.model.eval()
                pred, spatial_re, temporal_re, mu, sigma = self.model(target, temporal1, temporal2, temporal3, support1, support2, support3, static)
        return temporal_re

    def test(self, area, time, temporal_re):
        dataset = JD(7, area, time, mode='testing')
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        data_size = 0
        tr_loss = 0
        for batch_idx, (temporal1, temporal2, temporal3, support1, support2, support3, static, target) in enumerate(dataloader):
            data_size = target.shape[1]
            self.model.eval()
            spatial_re = self.model.spatial_representation(static)
            pred = self.model.generation(temporal_re, spatial_re, temporal1, temporal2, temporal3, support1, support2, support3, static, target)
            MSE = torch.nn.MSELoss(reduce=False, size_average=False)
            mse = MSE(target, pred)
            mse = mse.sum()
            tr_loss += mse.item()
        print('====> Test Average loss:', tr_loss / (len(dataloader.dataset) * data_size))
        return tr_loss / (len(dataloader.dataset) * data_size)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":

    # initial training args

    parser = argparse.ArgumentParser(description='PP Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')                                                              
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--initial_lr', type=int, default=0.0001, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
   
    # train
    areas = ['怀柔区', '崇文区', '门头沟', '昌平区', '大兴区', '朝阳区', '延庆区', '宣武区', '石景山区', '平谷区', '西城区', '海淀区', '房山区', '密云区', '东城区', '顺义区', '通州区', '丰台区']
    times = [0, 1, 618, 10, 11, 12]

    model = MST(**model_configs.MODEL_CONFIGS['jd'])

    train = Trainer(model, args)

    area_re = {}
    time_re = {}

    # area: all, single area
    # time: all, daily, weekend, holiday
    train.load('../data/mst.model')
    for time in times:
        print(time)
        temporal_re = train.generate_temporal(area='all', time=time)
        testing_loss = train.test('all', time, temporal_re)
    exit()
    
    while(1):
        # co-training bettween time and area
        try:
            for area in areas:
                print('spatial training!')
                # train_area = random.choice(areas) # random select a time to training
                train_area = area
                print('selected:', train_area)
                current_loss, spatial_re, temporal_re = train.train(area=train_area, time='all') # mode: spatial training
                area_re[area] = temporal_re

            for time in times:
                print('temporal training!')
                # train_time = random.choice(times) # random select a area to training
                train_time = time
                print('selected:', train_time)
                current_loss, spatial_re, temporal_re = train.train(area='all', time=train_time) # mode: temporal training
                time_re[time] = temporal_re
            
            for time in times:
                train_time = time
                print('selected:', train_time)
                testing_loss = train.test('all', train_time, time_re[time])

        except:
            train.save('../data/mst.model')
            exit()

