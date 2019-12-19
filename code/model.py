import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layer


class TemporalRepresentation(nn.Module):

    def __init__(self, order_feature=73, support_feature=73, embedding_out=50, sub_seq_in=50, sub_seq_out=40, \
                    mlp_in=40, mlp_out=30, tem_att_in=30, seq_in=30, seq_out=10, fea_att_in=10, \
                        fin_in=10, fin_out=5, sample_L=10, time_num=3):
        super(TemporalRepresentation, self).__init__()
        self.time_num = time_num

        self.L = sample_L
        # embedding data feature
        self.target_feature_embedding_layer = nn.Linear(order_feature, embedding_out)
        self.order_feature_embedding_layer = nn.Linear(order_feature, embedding_out)
        self.support_feature_embedding_layer = nn.Linear(support_feature, embedding_out)
        # purchase seq embedding
        self.lstm_order_recent = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        self.lstm_order_week = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        self.lstm_order_month = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        # shopping cart seq embedding
        self.lstm_support_recent = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        self.lstm_support_week = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        self.lstm_support_month = nn.LSTM(input_size=sub_seq_in, hidden_size=sub_seq_out, batch_first=True)
        # fusion different time feature
        self.temporal_order_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in, output_feature=mlp_out)
        self.temporal_support_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in, output_feature=mlp_out)
        # temporal attention
        self.attention_t_order = layer.Attention(dimensions=tem_att_in)
        self.attention_t_support = layer.Attention(dimensions=tem_att_in)
        # seq deal the attentioned data
        self.lstm_mixture_order = nn.LSTM(input_size=seq_in, hidden_size=seq_out, batch_first=True)
        self.lstm_mixture_support = nn.LSTM(input_size=seq_in, hidden_size=seq_out, batch_first=True)
        # feature attention
        self.attention_f = layer.Attention(dimensions=fea_att_in)
        # obtian the representation of temporal
        self.representation_layer_mu = nn.Linear(fin_in, fin_out)
        self.representation_layer_sigma = nn.Linear(fin_in, fin_out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, target_data, order_data_recent, order_data_week, order_data_month, \
                 support_data_recent, support_data_week, support_data_month):

        embedding_ord_recent = F.relu(self.order_feature_embedding_layer(order_data_recent))
        embedding_ord_week = F.relu(self.order_feature_embedding_layer(order_data_week))
        embedding_ord_month = F.relu(self.order_feature_embedding_layer(order_data_month))
        
        embedding_sup_recent = F.relu(self.support_feature_embedding_layer(support_data_recent))
        embedding_sup_week = F.relu(self.support_feature_embedding_layer(support_data_week))
        embedding_sup_month = F.relu(self.support_feature_embedding_layer(support_data_month))

        embedding_tar = F.relu(self.target_feature_embedding_layer(target_data))

        seq_ord_recent, (hn, cn) = self.lstm_order_recent(embedding_ord_recent)
        seq_ord_week, (hn, cn) = self.lstm_order_week(embedding_ord_week)
        seq_ord_month, (hn, cn) = self.lstm_order_month(embedding_ord_month)

        seq_ord_recent = F.relu(seq_ord_recent)
        seq_ord_week = F.relu(seq_ord_week)
        seq_ord_month = F.relu(seq_ord_month)

        seq_sup_recent, (hn, cn) = self.lstm_support_recent(embedding_sup_recent)
        seq_sup_week, (hn, cn) = self.lstm_support_week(embedding_sup_week)
        seq_sup_month, (hn, cn) = self.lstm_support_month(embedding_sup_month)

        seq_sup_recent = F.relu(seq_sup_recent)
        seq_sup_week = F.relu(seq_sup_week)
        seq_sup_month = F.relu(seq_sup_month)

        fusion_temporal_ord = torch.stack([seq_ord_recent, seq_ord_week, seq_ord_month])
        fusion_temporal_sup = torch.stack([seq_sup_recent, seq_sup_week, seq_sup_month])

        fusion_temporal_ord = F.relu(self.temporal_order_fusion_mlp(fusion_temporal_ord))
        fusion_temporal_sup = F.relu(self.temporal_support_fusion_mlp(fusion_temporal_sup))

        seq_ord, attention_weights_ord = self.attention_t_order(embedding_tar, fusion_temporal_ord)
        seq_sup, attention_weights_sup = self.attention_t_support(embedding_tar, fusion_temporal_sup)

        output, (fin_ord, cn) = self.lstm_mixture_order(seq_ord)
        output, (fin_sup, cn) = self.lstm_mixture_support(seq_sup)

        fin_ord = F.relu(fin_ord)
        fin_sup = F.relu(fin_sup)

        fusion_temporal = torch.cat([fin_ord, fin_sup])
        fusion_temporal = torch.stack([fin_ord, fin_sup]).permute(1,0,2)
        fusion_temporal, attention_weight_fusion = self.attention_f(embedding_tar, fusion_temporal)

        representation = F.relu(fusion_temporal)

        representation = torch.squeeze(representation)
        representation = torch.mean(representation, 0)

        mu = self.representation_layer_mu(representation)
        std = self.representation_layer_sigma(representation)

        representation = []
        for i in range(self.L):
            representation.append(self.reparameterize(mu, std))

        representation = torch.stack(representation)
        representation = torch.mean(representation, 0)

        representation = torch.squeeze(representation)

        return representation


class SpatioRepresentation(nn.Module):

    def __init__(self, order_feature=73, region_feature=13, embedding_out_tar=20, region_fea_list=[2,3,4,5], \
                    mlp_in=10, mlp_out=5, att_r=5, fc_out=20, fin_in=20, fin_out=5, region_num = 15, sample_L=10):
        super(SpatioRepresentation, self).__init__()
        self.L = sample_L
        self.region_num = region_num

        self.target_feature_embedding_layer = nn.Linear(order_feature, embedding_out_tar)

        self.mlp_regions = layer.MLP_s(input_nums=region_fea_list, input_feature=mlp_in, output_feature=mlp_out)

        self.attention_region = layer.Attention(att_r)

        self.region_fusion = nn.Linear(region_num, 1)

        self.feature_fusion = nn.Linear(mlp_out, fc_out)

        self.representation_layer_mu = nn.Linear(fin_in, fin_out)
        self.representation_layer_sigma = nn.Linear(fin_in, fin_out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, target_data, neighbor_region_data, current_region_data):
        embedding_tar = F.relu(self.target_feature_embedding_layer(target_data))

        regions = []
        for i in range(self.region_num):
            regions.append(F.relu(self.mlp_regions(neighbor_region_data[i])))
        regions = torch.stack(regions)
        # region_num * batch_size * region_feature -> batch * region_num * region_feature
        regions = regions.permute(1,0,2)
        
        att_regions, attention_weight_region = self.attention_region(embedding_tar, regions)

        # fusion multiple region into one representation
        # batch * region_num * region_feature -> batch * region_feature
        att_regions = att_regions.permute(0,2,1)
        att_regions = F.relu(self.region_fusion(att_regions))
        att_regions = torch.squeeze(att_regions.permute(0,2,1))

        region_representation = F.relu(self.feature_fusion(att_regions))
        representation = region_representation * embedding_tar
     
        representation = torch.mean(representation, 0)

        representation = torch.squeeze(representation)

        mu = self.representation_layer_mu(representation)
        std = self.representation_layer_sigma(representation)

        representation = []
        for i in range(self.L):
            representation.append(self.reparameterize(mu, std))

        representation = torch.stack(representation)
        representation = torch.mean(representation, 0)
        representation = torch.squeeze(representation)

        return representation


class Generation(nn.Module):

    def __init__(self, order_feature=73, support_feature=73, region_feature=13, G_input=5, G_hidden=[25,50], G_output=73, sample_L=10, time_num=3, \
                        embedding_out=50, sub_seq_in=50, sub_seq_out=40, mlp_in_t=40, mlp_out_t=30, \
                        tem_att_in=30, seq_in=30, seq_out=10, fea_att_in=10, fin_in_t=10, fin_out_t=5, \
                        embedding_out_tar=20, region_fea_list=[2,3,4,5], mlp_in_s=10, mlp_out_s=5, \
                        att_r=5, fc_out=20, fin_in_s=5, fin_out_s=5, region_num=15):
        
        super(Generation, self).__init__()

        self.time_num = time_num

        self.temporal_representation_layer = TemporalRepresentation(order_feature, support_feature, embedding_out, sub_seq_in, sub_seq_out, mlp_in_t, mlp_out_t, \
                                                                        tem_att_in, seq_in, seq_out, fea_att_in, fin_in_t, fin_out_t, sample_L, time_num)

        self.spatio_representation_layer = SpatioRepresentation(order_feature, region_feature, embedding_out_tar, region_fea_list, mlp_in_s, mlp_out_s, \
                                                                    att_r, fc_out, fin_in_s, fin_out_s, region_num, sample_L)

        self.support_recent_embedding_layer = nn.LSTM(support_feature, fin_out_t, num_layers=1, batch_first=True)
        self.support_week_embedding_layer = nn.LSTM(support_feature, fin_out_t, num_layers=1, batch_first=True)
        self.support_month_embedding_layer = nn.LSTM(support_feature, fin_out_t, num_layers=1, batch_first=True)
        self.temporal_order_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in_t, output_feature=mlp_out_t)
        self.support_embedding_layer = nn.LSTM(seq_in, seq_out, num_layers=1, batch_first=True)
        self.support_representation_layer = nn.Linear(seq_out, fin_out_t)

        self.order_recent_embedding_layer = nn.LSTM(order_feature, fin_out_t, num_layers=1, batch_first=True)
        self.order_week_embedding_layer = nn.LSTM(order_feature, fin_out_t, num_layers=1, batch_first=True)
        self.order_month_embedding_layer = nn.LSTM(order_feature, fin_out_t, num_layers=1, batch_first=True)
        self.temporal_support_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in_t, output_feature=mlp_out_t)
        self.order_embedding_layer = nn.LSTM(seq_in, seq_out, num_layers=1, batch_first=True)
        self.order_representation_layer = nn.Linear(seq_out, fin_out_t)

        self.mlp_regions = layer.MLP_s(input_nums=region_fea_list, input_feature=mlp_in_s, output_feature=mlp_out_s)
        self.region_fusion = nn.Linear(region_num, 1)
        self.region_representation_layer = nn.Linear(region_feature, fin_out_s)

        self.current_embedding_layer = nn.Linear(2*fin_out_t+fin_out_s, G_input)

        self.mix_net = [nn.Linear(G_input+fin_out_t+fin_out_s, G_hidden[0]), nn.ReLU()]
        for i in range(len(G_hidden)-1):
            self.mix_net.extend([nn.Linear(G_hidden[i], G_hidden[i+1]), nn.ReLU()])
        self.mix_net.extend([nn.Linear(G_hidden[-1], G_output), nn.ReLU()])
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

        