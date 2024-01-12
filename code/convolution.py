import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean
from robust_aggregation import ro_coefficient
import torch.nn as nn


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class Convolution(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_labels,
                 robust_aggr=False,  # whether to use robust aggregation
                 norm_embed=False,
                 bias=True):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.robust_aggr = robust_aggr
        self.norm_embed = norm_embed
        self.num_labels = num_labels
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.trans_weight = Parameter(torch.Tensor(self.num_labels, int(self.in_channels / 4)))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        size2 = self.trans_weight.size(0)

        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size2, self.trans_weight)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class ConvolutionBase_in_out(Convolution):
    """
    First layer of TrustGuard.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: node feature matrix.
        :param edge_index: Indices.
        :param edge_label: Edge attribute (i.e., trust relationship) vector.
        """
        row, col = edge_index  # row: trustor index  col: trustee index
        edge_label_trans = torch.matmul(edge_label, self.trans_weight)
        if not self.robust_aggr:
            opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
            # aggregating out-degree neighbors' node embedding
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

            inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
            # aggregating in-degree neighbors' node embedding
            inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            print('---------- Basic robust aggregation starts! ----------')
            row_out, col_out = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
            ro_out = ro_coefficient(x, 1.4, row_out, col_out)
            edge_label_trans_out = torch.mul(edge_label_trans, ro_out)
            x_out = torch.mul(x[col], ro_out)
            opinion = scatter_add(edge_label_trans_out, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x_out, row, dim=0, dim_size=x.size(0))

            row_in, col_in = edge_index[1].cpu().data.numpy()[:], edge_index[0].cpu().data.numpy()[:]
            ro_in = ro_coefficient(x, 1.4, row_in, col_in)
            edge_label_trans_in = torch.mul(edge_label_trans, ro_in)
            x_in = torch.mul(x[row], ro_in)
            inn_opinion = scatter_add(edge_label_trans_in, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x_in, col, dim=0, dim_size=x.size(0))

        out = torch.cat((out, opinion, inn, inn_opinion), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:  # False
            out = F.normalize(out, p=2, dim=-1)

        return out


class ConvolutionDeep_in_out(Convolution):
    """
    Deep layers of TrustGuard.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: Features from previous layer.
        :param edge_index: Indices.
        :param edge_label: Edge attribute (i.e., trust relationship) vector.
        :return out: Abstract convolved features.
        """
        row, col = edge_index
        edge_label_trans = torch.matmul(edge_label, self.trans_weight)
        if not self.robust_aggr:
            opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

            inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
            inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            print('---------- Deep robust aggregation starts! ----------')
            row_out, col_out = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
            ro_out = ro_coefficient(x, 1.4, row_out, col_out)
            edge_label_trans_out = torch.mul(edge_label_trans, ro_out)
            x_out = torch.mul(x[col], ro_out)
            opinion = scatter_add(edge_label_trans_out, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x_out, row, dim=0, dim_size=x.size(0))

            row_in, col_in = edge_index[1].cpu().data.numpy()[:], edge_index[0].cpu().data.numpy()[:]
            ro_in = ro_coefficient(x, 1.4, row_in, col_in)
            edge_label_trans_in = torch.mul(edge_label_trans, ro_in)
            x_in = torch.mul(x[row], ro_in)
            inn_opinion = scatter_add(edge_label_trans_in, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x_in, col, dim=0, dim_size=x.size(0))

        out = torch.cat((out, opinion, inn, inn_opinion), 1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out


class AttentionNetwork(nn.Module):
    """
    Position-aware self-attention mechanism.
    """
    def __init__(self, input_dim, n_heads, num_time_slots, attn_drop, residual):
        super(AttentionNetwork, self).__init__()
        self.n_heads = n_heads
        self.num_time_slots = num_time_slots
        self.residual = residual

        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_slots, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))

        self.lin = nn.Linear(input_dim, input_dim, bias=False)  # False is better (not test much)
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        # 1. add position embeddings
        position_inputs = torch.arange(0,self.num_time_slots).reshape(1,-1).repeat(inputs.shape[0],1).long()
        # [N,T]  [[0,1,2..,time-1],[0,1,2,..,time-1],...]
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N,T,F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0]))  # [N,T,F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0]))
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0]))

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0,2,1))  # [hN, T, T]
        outputs = outputs / (self.num_time_slots) ** 0.5  # scaling

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.zeros(self.num_time_slots-1, self.num_time_slots)
        one = torch.ones(1,self.num_time_slots)

        tril = torch.cat((diag_val,one),0)
        masks = tril.repeat(outputs.shape[0], 1, 1).to(inputs.device)  # [h*N, T, T]

        padding = torch.ones_like(masks) * (-2**32+1)  # [h*N, T, T]
        outputs = torch.where(masks==0, padding, outputs)  # if masks == 0, then padding
        outputs = F.softmax(outputs, dim=2)  # attention matrix: [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2)
        # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        # outputs = F.elu(self.lin(inputs))  # elu is worse than relu
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph convolutional network class.
    """
    def __init__(self, device, args, X, num_labels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.args = args
        torch.manual_seed(self.args.seed)  # fixed seed == 42
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Constructing trust propagation layers.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers  # layers = [32,64,32]
        self.layers = len(self.neurons)
        self.aggregators = []

        self.base_aggregator = ConvolutionBase_in_out(self.X.shape[1] * 4, self.neurons[0], self.num_labels).to(self.device)
        for i in range(1, self.layers):
            self.aggregators.append(
                ConvolutionDeep_in_out(self.neurons[i - 1] * 4, self.neurons[i], self.num_labels).to(self.device))

        self.aggregators = ListModule(*self.aggregators)

    def forward(self, train_edges, y_train):
        """
        Trust propagation and aggregation.
        return output: node embeddings at the last layer.
        """
        h = []
        self.X = F.dropout(self.X, self.dropout, training=self.training)
        h.append(torch.relu(self.base_aggregator(self.X, train_edges, y_train)))

        for i in range(1, self.layers):
            h[-1] = F.dropout(h[-1], self.dropout, training=self.training)
            h.append(torch.relu(self.aggregators[i - 1](h[i - 1], train_edges, y_train)))
        output = h[-1]

        return output
