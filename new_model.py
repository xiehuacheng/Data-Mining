import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from dgl.nn import MaxPooling

class GAT_1(nn.Module):
    def __init__(self, in_feature, edge_feature, num_heads, num_hidden, dropout, alpha):
        super(GAT_1, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        
        # 边输入层
        self.edge_input_layer1 = nn.Linear(edge_feature, 8)
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=in_feature, out_feats=num_hidden//2, norm='both', weight=True, bias=True, activation=None)
        self.conv2 = dglnn.GraphConv(in_feats=num_hidden//2, out_feats=num_hidden, norm='both', weight=True, bias=True, activation=None)
        
        # SAGEConv层
        self.sageconv1 = dglnn.SAGEConv(in_feats=num_hidden, out_feats=num_hidden, aggregator_type='lstm', feat_drop=0.0, activation=None, bias=True)
        
        # EdgeGATConv层
        self.edgegatconv1 = dglnn.EdgeGATConv(in_feats=num_hidden, edge_feats=8, out_feats=num_hidden, num_heads=num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=alpha, residual=True, activation=None, allow_zero_in_degree=True, bias=True)

        self.active_layer1 = nn.Linear(num_heads * num_hidden, 1)
        self.consume_layer1 = nn.Linear(num_heads * num_hidden, 1)

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        y = g.edata['feat']
        
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型
        y = y.to(torch.float32)  # 将输入张量转换为 float32 数据类型
        
        # 执行边输入层操作
        y = self.edge_input_layer1(y)
        y = F.relu(y)

        # 执行GCN卷积操作
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        x = F.relu(x)
        
        # 执行dropout操作
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 执行SAGEConv操作
        x = self.sageconv1(g, x)
        x = F.relu(x)
        
        # 执行dropout操作
        x = F.dropout(x, self.dropout, training=self.training)

        # 执行图卷积操作
        x = self.edgegatconv1(g, x, y)
        
        # 执行dropout操作
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 修改x的形状
        x = x.view(-1, self.num_heads * self.num_hidden)
        
        # 执行输出层操作
        active_index = self.active_layer1(x)
        consume_index = self.consume_layer1(x)
        
        active_index = F.relu(active_index)
        consume_index = F.relu(consume_index)
        
        return active_index, consume_index
    
class GCN_1(nn.Module):
    def __init__(self, in_feature):
        super(GCN_1, self).__init__()
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=in_feature, out_feats=64, norm='both', weight=True, bias=True, activation=None)
        self.conv2 = dglnn.GraphConv(in_feats=64, out_feats=256, norm='both', weight=True, bias=True, activation=None)
        self.conv3 = dglnn.GraphConv(in_feats=256, out_feats=512, norm='both', weight=True, bias=True, activation=None)
        self.conv4 = dglnn.GraphConv(in_feats=512, out_feats=512, norm='both', weight=True, bias=True, activation=None)

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        
        # 连接层
        self.shortcut1 = nn.Linear(in_feature, 64)
        self.shortcut2 = nn.Linear(64, 256)
        self.shortcut3 = nn.Linear(256, 512)
        
        # 输出层
        self.active_layer1 = nn.Linear(512, 128)
        self.active_layer2 = nn.Linear(128, 1)
        self.consume_layer1 = nn.Linear(512, 128)
        self.consume_layer2 = nn.Linear(128, 1)

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型

        # 执行GCN卷积操作
        x1 = self.conv1(g, x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1 + self.shortcut1(x))  # 使用线性变换改变x的维度

        x2 = self.conv2(g, x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.shortcut2(x1))  # 使用线性变换改变x1的维度

        x3 = self.conv3(g, x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + self.shortcut3(x2))  # 使用线性变换改变x2的维度

        x4 = self.conv4(g, x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4 + x3)  # 使用线性变换改变x3的维度
        
        # 执行输出层操作
        active_index = self.active_layer1(x4)
        consume_index = self.consume_layer1(x4)

        active_index = F.relu(active_index)
        consume_index = F.relu(consume_index)
        
        # active_index = F.dropout(active_index, 0.1, training=self.training)
        # consume_index = F.dropout(consume_index, 0.1, training=self.training)

        active_index = self.active_layer2(active_index)
        consume_index = self.consume_layer2(consume_index)
        
        return active_index, consume_index
    
class GAT_2(nn.Module):
    def __init__(self, in_feature):
        super(GAT_2, self).__init__()
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=in_feature, out_feats=64, norm='both', weight=True, bias=True, activation=None)
        self.conv2 = dglnn.GraphConv(in_feats=64, out_feats=256, norm='both', weight=True, bias=True, activation=None)
        self.conv3 = dglnn.GraphConv(in_feats=256, out_feats=512, norm='both', weight=True, bias=True, activation=None)
        self.conv4 = dglnn.GraphConv(in_feats=512, out_feats=512, norm='both', weight=True, bias=True, activation=None)

        # GAT层
        self.gat1 = dglnn.GATConv(in_feats=512, out_feats=128, num_heads=8, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=True, activation=None, allow_zero_in_degree=True, bias=True)

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=1024)
        
        # 连接层
        self.shortcut1 = nn.Linear(in_feature, 64)
        self.shortcut2 = nn.Linear(64, 256)
        self.shortcut3 = nn.Linear(256, 512)
        self.shortcut4 = nn.Linear(512, 512)
        self.shortcut5 = nn.Linear(512, 1024)
        
        # Flatten层
        self.flatten = nn.Flatten()
        
        # 输出层
        self.active_layer1 = nn.Linear(1024, 128)
        self.active_layer2 = nn.Linear(128, 1)
        self.consume_layer1 = nn.Linear(1024, 128)
        self.consume_layer2 = nn.Linear(128, 1)

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型

        # 执行GCN卷积操作
        x1 = self.conv1(g, x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1 + self.shortcut1(x))  # 使用线性变换改变x的维度

        x2 = self.conv2(g, x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.shortcut2(x1))  # 使用线性变换改变x1的维度

        x3 = self.conv3(g, x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + self.shortcut3(x2))  # 使用线性变换改变x2的维度

        x4 = self.conv4(g, x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4 + x3)  # 使用线性变换改变x3的维度
        
        # 执行GAT卷积操作
        x5 = self.gat1(g, x4)
        x5 = self.flatten(x5)
        x5 = self.bn5(x5)
        x5 = F.relu(x5 + self.shortcut5(x4))  # 使用线性变换改变x4的维度
        
        # 执行输出层操作
        active_index = self.active_layer1(x5)
        consume_index = self.consume_layer1(x5)

        active_index = F.relu(active_index)
        consume_index = F.relu(consume_index)
        
        # active_index = F.dropout(active_index, 0.1, training=self.training)
        # consume_index = F.dropout(consume_index, 0.1, training=self.training)

        active_index = self.active_layer2(active_index)
        consume_index = self.consume_layer2(consume_index)
        
        return active_index, consume_index

class GAT_3(nn.Module):
    def __init__(self, in_feature):
        super(GAT_3, self).__init__()
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=in_feature, out_feats=64, norm='both', weight=True, bias=True, activation=torch.nn.ReLU())
        self.conv2 = dglnn.GraphConv(in_feats=64, out_feats=256, norm='both', weight=True, bias=True, activation=torch.nn.ReLU())
        self.conv3 = dglnn.GraphConv(in_feats=256, out_feats=512, norm='both', weight=True, bias=True, activation=torch.nn.ReLU())
        # self.conv4 = dglnn.GraphConv(in_feats=512, out_feats=512, norm='both', weight=True, bias=True, activation=torch.nn.ReLU())
        
        # GAT层
        self.gat1 = dglnn.EdgeGATConv(in_feats=512, edge_feats=2, out_feats=128, num_heads=8, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=True, activation=None, allow_zero_in_degree=True, bias=True)
        
        # Flatten层
        self.flatten = nn.Flatten()

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        
        self.bn5 = nn.BatchNorm1d(num_features=1024)
        
        # 输出层
        self.active_layer1 = nn.Linear(1024, 128)
        self.active_layer2 = nn.Linear(128, 1)
        self.consume_layer1 = nn.Linear(1024, 128)
        self.consume_layer2 = nn.Linear(128, 1)

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        y = g.edata['feat']
        
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型
        y = y.to(torch.float32)  # 将输入张量转换为 float32 数据类型

        # 执行GCN卷积操作
        x = self.conv1(g, x)
        x = self.bn1(x)

        x = self.conv2(g, x)
        x = self.bn2(x)

        x = self.conv3(g, x)
        x = self.bn3(x)

        # x = self.conv4(g, x)
        # x = self.bn4(x)
        
        # 执行GAT卷积操作
        x = self.gat1(g, x, y)
        x = self.flatten(x)
        x = self.bn5(x)
        
        # 执行输出层操作
        active_index = self.active_layer1(x)
        consume_index = self.consume_layer1(x)

        active_index = F.relu(active_index)
        consume_index = F.relu(consume_index)

        active_index = self.active_layer2(active_index)
        consume_index = self.consume_layer2(consume_index)
        
        return active_index, consume_index