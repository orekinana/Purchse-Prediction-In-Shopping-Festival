import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='no'):
        super(Attention, self).__init__()

        self.attn = nn.Linear(dimensions, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.attention_type = attention_type

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, dimensions]):
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.   
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, query length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, query length]):
              Tensor containing attention weights.
        """

        # (batch_size, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, query_len)
        if self.attention_type == 'seq':
            attention_scores = []
            for batch in range(query.shape[0]):
                attention_scores.append(query[batch] * context[batch])
            attention_scores = torch.stack(attention_scores)
        else:
            attention_scores = torch.mul(query , context)

        attention_weights = self.softmax(attention_scores)

        # (batch_size, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, query_len, dimensions)
        mix = torch.mul(attention_weights, context)

        # output -> (batch_size, query_len, dimensions)
        output = self.attn(mix)
        output = self.tanh(output)

        return output, attention_weights


class MLP(nn.Module):
    # inputs_shape: [inputs number, input feature number]
    def __init_(self, inputs_shape):
        super(MLP, self).__init__()
        self.input_num = inputs_shape[0]
        self.feature_num = inputs_shape[1]
        self.fc_list = [nn.Linear(self.input_num, 1) for i in range(self.feature_num)]

    # inputs is a numpy array with shape (inputs number * input feature number)
    def forward(self, inputs):
        output = []
        for i in range(self.feature_num):
            output.append(F.relu(self.fc_list[i](inputs[:, i])))
        return np.array(output)


class Pooling(nn.Module):

    def __init__(self, pooling_type='mean'):
        super(Pooling, self).__init__()
        self.type = pooling_type

    def forward(self, hidden_data):
        output = hidden_data
        if self.pooling_type == 'mean':
            output = torch.mean(hidden_data)
        return output