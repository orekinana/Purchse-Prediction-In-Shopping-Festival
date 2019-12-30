import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TemporalRepresentation as TR
from model import SpatioRepresentation as SR
from model import Generation as G
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
        dataset = JD(7, area, time)
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        data_size = 0
        total_tr_loss = 0
        for epoch in range(self.args.epochs):
            tr_loss = 0
            for batch_idx, (temporal, spatio, target) in enumerate(self.dataloader):
                print('target shape:', target.shape)
                data_size = target.shape[1]
                self.model.train()
                pred = self.model(target, temporal, spatio)
                loss = self.model.loss_function(target, pred)
                loss = loss.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(target), \
                                                            len(self.dataloader.dataset), 100. * batch_idx / len(self.dataloader.dataset), loss.item() / target.shape[0] / data_size))
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, tr_loss / (len(self.dataloader.dataset) * data_size)))
            total_tr_loss += tr_loss
        total_tr_loss /= self.args.epochs
        return total_tr_loss
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":

    # initial training args

    parser = argparse.ArgumentParser(description='PP Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

    model = G(**model_configs.MODEL_CONFIGS['jd'])
    train = Trainer(model, args)

    # area: all, single area
    # time: all, daily, weekend, holiday
    threshold = 0.00001
    while(1):
        # co-training bettween time and area
        last_loss = 0
        area_cycle = 0
        while(1):
            train_time = random.choice(times) # random select a area to training
            current_loss = train.train(area='all', time=train_time) # mode: temporal training
            if abs(current_loss-last_loss) < threshold:
                break
            last_loss = current_loss
            area_cycle += 1

        last_loss = 0
        time_cycle = 0
        while(1):
            train_area = random.choice(areas) # random select a time to training
            current_loss = train.train(area=train_area, time='all') # mode: spatial training
            if abs(current_loss-last_loss) < threshold:
                break
            last_loss = current_loss
            time_cycle += 1
        
        if area_cycle == 0 and time_cycle == 0:
            break