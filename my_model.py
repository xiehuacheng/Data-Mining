import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 定义可学习参数
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # 输入说明：
        # h: 节点特征矩阵 (N, in_features)
        # adj: 邻接矩阵 (N, N)

        # 为了方便后续的计算，我们将邻接矩阵 adj 转换为稠密矩阵
        adj = torch.squeeze(adj, -1)

        # 计算 Wh = h * W
        Wh = torch.matmul(h, self.W)  # (N, out_features)

        # Wh1 + Wh2.T 是N * N矩阵，第i行第j列是 Wh1[i] + Wh2[j]
        # 那么 Wh1 + Wh2.T 的第i行第j列刚好就是文中的 a^T * [Whi||Whj]
        # 代表着节点i对节点j的attention
        # 分成两个部分，分别用于计算注意力权重
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)

        # 计算注意力系数 e = leakyrelu(Wh1) + leakyrelu(Wh2).T
        e = self.leakyrelu(Wh1 + torch.transpose(Wh2, 2, 1))  # (N, N)

        # 创建一个填充矩阵，用于在 adj 为零的位置上进行填充
        # 仅保留邻接矩阵 adj 中非零的元素，并将其替换为 e 中相应的元素
        # 这意味着只有有连接的节点之间才会有非零的注意力系数
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)

        # 仅保留 adj 中非零的元素，并将其替换为 e 中相应的元素
        attention = torch.where(adj > 0, e, padding)  # (N, N)

        # 使用 softmax 函数对 attention 进行归一化，得到每个节点的注意力权重
        attention = F.softmax(attention, dim=1)  # (N, N)

        # 对注意力权重也做 dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 计算 h_prime = attention * Wh
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, date_emb, nfeat, nhid, dropout, alpha, nheads):
        '''
        日期嵌入(date_emb);
        输入特征维度(nfeat);
        隐藏层维度(nhid);
        丢弃率(dropout);
        LeakyReLU 激活函数的负斜率(alpha);
        多头注意力的数量(nheads);
        '''
        super(GAT, self).__init__()
        date_index_number, date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout

        # 创建多头注意力层
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        # 输出注意力层
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout, alpha, concat=False)
        
        # 日期嵌入层
        self.date_embedding = nn.Embedding(date_index_number, date_dim)
        
        # 输出层
        self.active_index = nn.Linear(nhid, 1)
        self.consume_index = nn.Linear(nhid, 1)
        
    def forward(self, x_date, x_feature, x_mask_data):
        x = x_feature
        
        # 经过多头注意力层
        x = torch.cat([head(x, x_mask_data) for head in self.MH], dim=-1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)
        
        # 经过输出注意力层
        x = self.out_att(x, x_mask_data)
        
        # 经过输出层
        act_pre = self.active_index(x)
        con_pre = self.consume_index(x)
        
        return act_pre, con_pre