import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layer


class TemporalRepresentation(nn.Module):

    def __init__(self, order_feature=30, support_feature=30, embedding_out=50, sub_seq_in=50, sub_seq_out=40, \
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

        fin_ord = torch.squeeze(F.relu(fin_ord))
        fin_sup = torch.squeeze(F.relu(fin_sup))

        fusion_temporal = torch.cat([fin_ord, fin_sup], dim=1)
        
        # fusion_temporal = torch.stack([fin_ord, fin_sup])
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

    def __init__(self, order_feature=30, region_feature=13, embedding_out_tar=20, region_fea_list=[23,27,4], \
                    mlp_in=3, mlp_out=5, att_r=5, fc_out=20, fin_in=5, fin_out=5, region_num = 15, sample_L=10):
        super(SpatioRepresentation, self).__init__()

        self.mlp_regions = layer.MLP_s(input_nums=region_fea_list, input_feature=mlp_in, output_feature=mlp_out)

        self.representation_region_layer = nn.Linear(fin_in, fin_out)


    
    def forward(self, target_data, region_data):
        embedding_region = F.relu(self.mlp_regions(region_data))
        representation = F.relu(self.representation_region_layer(embedding_region))

        return representation


class Generation(nn.Module):

    def __init__(self, order_feature=30, support_feature=30, region_feature=13, G_input=5, G_hidden=[25,50], G_output=30, sample_L=10, time_num=3, \
                        embedding_out=50, sub_seq_in=50, sub_seq_out=40, mlp_in_t=40, mlp_out_t=30, \
                        tem_att_in=30, seq_in=30, seq_out=10, fea_att_in=10, fin_in_t=10, fin_out_t=5, \
                        embedding_out_tar=20, region_fea_list=[23,27,4], mlp_in_s=10, mlp_out_s=5, \
                        att_r=5, fc_out=20, fin_in_s=5, fin_out_s=5, region_num=15):
        
        super(Generation, self).__init__()

        self.time_num = time_num

        self.temporal_representation_layer = TemporalRepresentation(order_feature, support_feature, embedding_out, sub_seq_in, sub_seq_out, mlp_in_t, mlp_out_t, \
                                                                        tem_att_in, seq_in, seq_out, fea_att_in, fin_in_t, fin_out_t, sample_L, time_num)

        self.spatio_representation_layer = SpatioRepresentation(order_feature, region_feature, embedding_out_tar, region_fea_list, mlp_in_s, mlp_out_s, \
                                                                    att_r, fc_out, fin_in_s, fin_out_s, region_num, sample_L)

        self.support_recent_embedding_layer = nn.LSTM(support_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.support_week_embedding_layer = nn.LSTM(support_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.support_month_embedding_layer = nn.LSTM(support_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.temporal_support_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in_t, output_feature=mlp_out_t)
        self.support_embedding_layer = nn.LSTM(seq_in, seq_out, num_layers=1, batch_first=True)
        self.support_representation_layer = nn.Linear(seq_out, fin_out_t)

        self.order_recent_embedding_layer = nn.LSTM(order_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.order_week_embedding_layer = nn.LSTM(order_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.order_month_embedding_layer = nn.LSTM(order_feature, sub_seq_out, num_layers=1, batch_first=True)
        self.temporal_order_fusion_mlp = layer.MLP_t(input_num=self.time_num, input_feature=mlp_in_t, output_feature=mlp_out_t)
        self.order_embedding_layer = nn.LSTM(seq_in, seq_out, num_layers=1, batch_first=True)
        self.order_representation_layer = nn.Linear(seq_out, fin_out_t)

        self.mlp_regions = layer.MLP_s(input_nums=region_fea_list, input_feature=mlp_in_s, output_feature=mlp_out_s)
        self.representation_region_layer = nn.Linear(fin_in_s, fin_out_s)

        self.current_embedding_layer = nn.Linear(2*fin_out_t+fin_out_s, G_input)

        self.mix_net = [nn.Linear(G_input+fin_out_t+fin_out_s, G_hidden[0]), nn.ReLU()]
        for i in range(len(G_hidden)-1):
            self.mix_net.extend([nn.Linear(G_hidden[i], G_hidden[i+1]), nn.ReLU()])
        self.mix_net.extend([nn.Linear(G_hidden[-1], G_output), nn.ReLU()])
        self.mix_net = torch.nn.Sequential(*self.mix_net)

    
    def forward(self, order_data_recent, order_data_week, order_data_month, support_data_recent, support_data_week, support_data_month, region_data, target_data):
        
        # current order and support embedding with shape batch * seqlen * feature
        support_recent, (hn, cn) = self.support_recent_embedding_layer(support_data_recent)
        support_week, (hn, cn) = self.support_week_embedding_layer(support_data_week)
        support_month, (hn, cn) = self.support_month_embedding_layer(support_data_month)

        fusion_temporal_sup = torch.stack([support_recent, support_week, support_month])
        fusion_temporal_sup = F.relu(self.temporal_support_fusion_mlp(fusion_temporal_sup))

        output, (support_embedding, cn) = self.support_embedding_layer(fusion_temporal_sup)  
        support_representation = self.support_representation_layer(support_embedding)
        support_representation = torch.squeeze(support_representation)

        order_recent, (hn, cn) = self.order_recent_embedding_layer(order_data_recent)
        order_week, (hn, cn) = self.order_week_embedding_layer(order_data_week)
        order_month, (hn, cn) = self.order_month_embedding_layer(order_data_month)

        fusion_temporal_order = torch.stack([order_recent, order_week, order_month])
        fusion_temporal_order = F.relu(self.temporal_order_fusion_mlp(fusion_temporal_order))

        output, (order_embedding, cn) = self.order_embedding_layer(fusion_temporal_order)  
        order_representation = self.order_representation_layer(order_embedding)
        order_representation = torch.squeeze(order_representation)

        embedding_region = F.relu(self.mlp_regions(region_data))
        region_representation = F.relu(self.representation_region_layer(embedding_region))

        current_representation = torch.cat([order_representation, support_representation, region_representation], dim=1)

        current_representation = self.current_embedding_layer(current_representation)

        # obtain the temporal and spatial representation of the current region and time slot
        spatio_representation = self.spatio_representation_layer(target_data, region_data)

        temporal_representation = self.temporal_representation_layer(target_data, order_data_recent, order_data_week, order_data_month, \
                 support_data_recent, support_data_week, support_data_month)

        # spatio_representation = torch.stack([spatio_representation for i in range(current_representation.shape[0])])
        temporal_representation = torch.stack([temporal_representation for i in range(current_representation.shape[0])])
        
        # concatanate the current feature and spati0-temporal representation of the current region and time slot
        mix_re = torch.cat([spatio_representation, temporal_representation, current_representation], dim=1)
        # predict the current of the data
        pred_data = self.mix_net(mix_re)
        return pred_data

    def loss_function(self, target_data, pred_data):
        MSE = torch.nn.MSELoss(reduce=False, size_average=False)
        
        return MSE(target_data, pred_data)

        