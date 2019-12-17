import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layer


class TemporalRepresentation(nn.Module):

    def __init__(self, order_feature=73, support_feature=73, out_feature=5, hidden_features={}, sample_L=10):
        super(TemporalRepresentation, self).__init__()

        self.L = sample_L
        # embedding data feature
        self.target_feature_embedding_layer = nn.Linear(order_feature, hidden_features['concate'])
        self.order_feature_embedding_layer = nn.Linear(order_feature, hidden_features['feature'])
        self.support_feature_embedding_layer = nn.Linear(support_feature, hidden_features['feature'])
        # seq deal
        self.lstm_order = nn.LSTM(input_size=hidden_features['feature'], hidden_size=hidden_features['feature'], batch_first=True) 
        self.lstm_support = nn.LSTM(input_size=hidden_features['feature'], hidden_size=hidden_features['feature'], batch_first=True)
        # concante seqence data and attention to target
        self.concate_layer = nn.Linear(hidden_features['feature'], hidden_features['feature'])
        self.attention = layer.Attention(dimensions=hidden_features['feature'], attention_type='seq')
        # seq deal the attentioned data
        self.lstm_mixture = nn.LSTM(input_size=hidden_features['concate'], hidden_size=hidden_features['representation'], batch_first=True)
        # obtian the representation of temporal
        self.representation_layer_mu = nn.Linear(hidden_features['representation'], out_feature)
        self.representation_layer_sigma = nn.Linear(hidden_features['representation'], out_feature)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, target_data, order_data, support_data=None):
        embedding_ord = F.relu(self.order_feature_embedding_layer(order_data))
        embedding_tar = F.relu(self.target_feature_embedding_layer(target_data))

        seq_ord, (hn, cn) = self.lstm_order(embedding_ord)
        # print(seq_ord.shape, hn.shape, cn.shape)
        seq_ord = F.relu(seq_ord)

        if support_data == None:
            seq_fea = seq_ord
        else:
            embedding_sup = F.relu(self.support_feature_embedding_layer(support_data)) 
            seq_sup, hn, cn = F.relu(self.lstm_support(embedding_sup))
            seq_fea= F.relu(self.concate_layer(torch.cat((seq_ord, seq_sup), dim=2)))

        output, attention_weights = self.attention(embedding_tar, seq_fea)
        att_representation = F.relu(output)

        output, (seq_mix, cn) = self.lstm_mixture(att_representation)
        embedding_representation = torch.squeeze(seq_mix)
        embedding_representation = torch.mean(embedding_representation, 0)

        mu = self.representation_layer_mu(embedding_representation)
        std = self.representation_layer_sigma(embedding_representation)

        representation = []
        for i in range(self.L):
            representation.append(self.reparameterize(mu, std))

        representation = torch.stack(representation)
        representation = torch.mean(representation, 0)

        representation = torch.squeeze(representation)

        return representation


class SpatioRepresentation(nn.Module):

    def __init__(self, order_feature=73, region_feature=13, hidden_features=[10,5], out_feature=5, sample_L=10):
        super(SpatioRepresentation, self).__init__()

        self.net = [nn.Linear(region_feature, hidden_features[0]), nn.ReLU()]
        for i in range(len(hidden_features)-1):
            self.net.extend([nn.Linear(hidden_features[i], hidden_features[i+1]), nn.ReLU()])
        self.net.extend([nn.Linear(hidden_features[-1], out_feature), nn.ReLU()])
        self.net = torch.nn.Sequential(*self.net)

        self.target_embedding_layer = nn.Linear(order_feature, out_feature)

        self.attention = layer.Attention(dimensions=out_feature)

        self.mix_layer = nn.ReLU(nn.Linear(out_feature, out_feature))

        self.representation_layer_mu = nn.Linear(out_feature, out_feature)
        self.representation_layer_sigma = nn.Linear(out_feature, out_feature)

        self.L = sample_L

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, target_data, feature_data):
        embedding_fea = self.net(feature_data)
        embedding_tar = self.target_embedding_layer(target_data)
        # print(embedding_tar.shape, embedding_fea.shape)

        att_embedding, attention_weights = self.attention(embedding_tar, embedding_fea)
        att_embedding = F.relu(att_embedding)

        embedding_representation = self.mix_layer(att_embedding)
        embedding_representation = torch.mean(embedding_representation, 0)

        embedding_representation = torch.squeeze(embedding_representation)

        mu = self.representation_layer_mu(embedding_representation)
        std = self.representation_layer_sigma(embedding_representation)

        representation = []
        for i in range(self.L):
            representation.append(self.reparameterize(mu, std))

        representation = torch.stack(representation)
        representation = torch.mean(representation, 0)
        representation = torch.squeeze(representation)

        return representation



class Generation(nn.Module):

    def __init__(self, order_feature=73, POIs_feature=10, region_feature=13, \
                         G_hidden_features=[50,30,10], S_hidden_features=[10,5], T_hidden_features={}, \
                            G_output_features=5, S_output_features=5, T_output_features=5, \
                                support_feature=73, sample_L=10):
        
        super(Generation, self).__init__()

        self.spatio_representation_layer = SpatioRepresentation(order_feature, region_feature, S_hidden_features, S_output_features, sample_L)

        self.temporal_representation_layer = TemporalRepresentation(order_feature, support_feature, T_output_features, T_hidden_features, sample_L)

        self.support_feature_embedding_layer = nn.LSTM(support_feature, T_output_features, num_layers=1, batch_first=True)
        self.order_feature_embedding_layer = nn.LSTM(order_feature, T_output_features, num_layers=1, batch_first=True)
        self.region_embedding_layer = nn.Linear(region_feature, S_output_features)
        self.poi_embedding_layer = nn.Linear(POIs_feature, S_output_features)

        self.current_embedding_layer = nn.Linear(2*(T_output_features+S_output_features), G_hidden_features[0])
        self.current_embedding_layer_without_pois_and_support = nn.Linear(T_output_features+S_output_features, G_hidden_features[0])

        self.mix_net = [nn.Linear(T_output_features + S_output_features + G_hidden_features[0], G_hidden_features[0]), nn.ReLU()]
        # print(T_output_features + S_output_features + G_hidden_features[0], G_hidden_features[0])
        for i in range(len(G_hidden_features)-1):
            self.mix_net.extend([nn.Linear(G_hidden_features[i], G_hidden_features[i+1]), nn.ReLU()])
        self.mix_net.extend([nn.Linear(G_hidden_features[-1], G_output_features), nn.ReLU()])
        self.mix_net = torch.nn.Sequential(*self.mix_net)

    
    def forward(self, target_data, order_data, region_data, POIs_data = None, support_data=None):
        
        # current order embedding with shape batch * seqlen * feature
        output, (order_embedding, cn) = self.order_feature_embedding_layer(order_data)
        # the final LSTM hidden output of current order embedding with shape batch * feature
        order_embedding = torch.squeeze(F.relu(order_embedding))

        #current region embedding with shape batch * feature
        region_embedding = F.relu(self.region_embedding_layer(region_data))
        if POIs_data != None:
            poi_embedding = F.relu(self.poi_embedding_layer(POIs_data))

        # concatenate the temporal and spatial feature
        if support_data == None and POIs_data == None:
            current_embedding = torch.cat([order_embedding, region_embedding], dim=1)
            current_embedding = F.relu(self.current_embedding_layer_without_pois_and_support(current_embedding))
        else:
            output, support_embedding, cn = self.support_feature_embedding_layer(support_data)
            support_embedding = F.relu(support_embedding)
            support_embedding = torch.squeeze(support_embedding)

            current_embedding = torch.cat([support_embedding, order_embedding, region_embedding, poi_embedding])
            current_embedding = F.relu(self.current_embedding_layer(current_embedding))

        # obtain the temporal and spatial representation of the current region and time slot
        spatio_re = self.spatio_representation_layer(target_data, region_data)
        temporal_re = self.temporal_representation_layer(target_data, order_data)

        spatio_re = torch.stack([spatio_re for i in range(current_embedding.shape[0])])
        temporal_re = torch.stack([temporal_re for i in range(current_embedding.shape[0])])
        
        # concatanate the current feature and spati0-temporal representation of the current region and time slot
        mix_re = torch.cat([spatio_re, temporal_re, current_embedding], dim=1)
        # predict the current of the data
        pred_data = self.mix_net(mix_re)
        return pred_data

    def loss_function(self, target_data, pred_data):
        MSE = torch.nn.MSELoss(reduce=False, size_average=False)
        
        return MSE(target_data, pred_data)

        