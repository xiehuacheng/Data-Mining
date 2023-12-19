import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
   
class GCN_LSTM(nn.Module):
    def __init__(self):
        super(GCN_LSTM, self).__init__()
        
        # Embedding Dim
        embed_dim = 10
                
        # NodeEmbedding层
        self.node_embedding = nn.Embedding(1140, embed_dim)
        
        # 定义参数
        gcn_in_feats1 = 35 + embed_dim
        gcn_out_feats1 = 128
        gcn_in_feats2 = gcn_out_feats1
        gcn_out_feats2 = 64
        
        lstm_in_feats1 = gcn_out_feats2
        lstm_out_feats1 = 32
        
        linear_in_feats1 = lstm_out_feats1 * 2
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=gcn_in_feats1, out_feats=gcn_out_feats1, norm='both', weight=True, bias=True, activation=None)
        self.conv2 = dglnn.GraphConv(in_feats=gcn_in_feats2, out_feats=gcn_out_feats2, norm='both', weight=True, bias=True, activation=None)
                
        # LSTM层
        self.lstm1 = nn.LSTM(input_size=lstm_in_feats1, hidden_size=lstm_out_feats1, num_layers=1, dropout=0.0, batch_first=False, bidirectional=True)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.1)

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(num_features=gcn_in_feats2)
        self.bn2 = nn.BatchNorm1d(num_features=lstm_in_feats1)
        self.bn3 = nn.BatchNorm1d(num_features=linear_in_feats1)
        
        # 连接层
        self.shortcut1 = nn.Linear(gcn_in_feats1, gcn_in_feats2)
        self.shortcut2 = nn.Linear(gcn_in_feats2, lstm_in_feats1)
        self.shortcut3 = nn.Linear(lstm_in_feats1, linear_in_feats1)
        
        # 输出层
        self.active_layer1 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.active_layer2 = nn.Linear(linear_in_feats1, 1)
        self.consume_layer1 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.consume_layer2 = nn.Linear(linear_in_feats1, 1)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        node_id = g.ndata['node_id']
                
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型
        
        # if self.training:
        #     # 给x添加噪声
        #     x = x + (torch.randn(x.size()) * 0.1).to(device='cuda:0')
        #     # 对x进行随机缩放，缩放比例为0.8~1.2
        #     x = x * (torch.rand(x.size()) * 0.2 + 0.9).to(device='cuda:0')
            
        # 执行NodeEmbedding操作
        node_id = self.node_embedding(node_id.long())
        
        # 转换node_id的数据类型
        node_id = node_id.to(torch.float32)
        
        # 将Embedding的结果作为输入特征的一部分
        x = torch.cat((x, node_id), 1)

        # 执行GCN卷积操作
        x1 = self.conv1(g, x)
        x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        x1 = self.activation(x1 + self.shortcut1(x))

        x2 = self.conv2(g, x1)
        x2 = self.bn2(x2)
        x2 = self.dropout(x2)
        x2 = self.activation(x2 + self.shortcut2(x1))
        
        # 执行LSTM操作
        x3, (h_n, c_n) = self.lstm1(x2)
        x3 = self.bn3(x3)
        x3 = self.dropout(x3)
        x3 = self.activation(x3 + self.shortcut3(x2))
        
        # 执行输出层操作
        active = self.active_layer1(x3)
        consume = self.consume_layer1(x3)
        
        active = self.activation(active)
        consume = self.activation(consume)
        
        active = self.active_layer2(active)
        consume = self.consume_layer2(consume)
        
        return active, consume

class GCN_LSTM_test(nn.Module):
    def __init__(self, time_step):
        super(GCN_LSTM_test, self).__init__()
        
        self.time_step = time_step
        
        # Embedding Dim
        embed_dim = 10
                
        # NodeEmbedding层
        self.node_embedding = nn.Embedding(1140, embed_dim)
        
        # 定义参数
        gcn_in_feats1 = 43 + embed_dim
        gcn_out_feats1 = 48
        gcn_in_feats2 = gcn_out_feats1
        gcn_out_feats2 = 24
        
        lstm_in_feats1 = gcn_out_feats2
        lstm_out_feats1 = 64
        
        linear_in_feats1 = lstm_out_feats1 * 4 + gcn_in_feats1
        
        # GraphConv层
        self.conv1 = dglnn.GraphConv(in_feats=gcn_in_feats1, out_feats=gcn_out_feats1, norm='both', weight=True, bias=True, activation=None)
        self.conv2 = dglnn.GraphConv(in_feats=gcn_in_feats2, out_feats=gcn_out_feats2, norm='both', weight=True, bias=True, activation=None)
                
        # LSTM层
        self.lstm1 = nn.LSTM(input_size=lstm_in_feats1, hidden_size=lstm_out_feats1, num_layers=2, dropout=0.1, batch_first=False, bidirectional=True)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.1)

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(num_features=gcn_in_feats2)
        self.bn2 = nn.BatchNorm1d(num_features=lstm_in_feats1)
        self.bn3 = nn.BatchNorm1d(num_features=linear_in_feats1)
        
        # 输出层
        self.active_layer1 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.active_layer2 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.active_layer3 = nn.Linear(linear_in_feats1, 1)
        self.consume_layer1 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.consume_layer2 = nn.Linear(linear_in_feats1, linear_in_feats1)
        self.consume_layer3 = nn.Linear(linear_in_feats1, 1)
        
        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, g):
        
        g = dgl.add_self_loop(g)
        
        x = g.ndata['feat']
        node_id = g.ndata['node_id']
                
        x = x.to(torch.float32)  # 将输入张量转换为 float32 数据类型
        
        # if self.training:
        #     # 给x添加噪声
        #     x = x + (torch.randn(x.size()) * 0.1).to(device='cuda:0')
        #     # 对x进行随机缩放，缩放比例为0.8~1.2
        #     # x = x * (torch.rand(x.size()) * 0.4 + 0.8).to(device='cuda:0')
            
        # 执行NodeEmbedding操作
        node_id = self.node_embedding(node_id.long())
        
        # 转换node_id的数据类型
        node_id = node_id.to(torch.float32)
        
        # 将Embedding的结果作为输入特征的一部分
        x = torch.cat((x, node_id), 1)
        
        # print(x.shape)

        # 执行GCN卷积操作
        x1 = self.conv1(g, x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        
        # print(x1.shape)

        x2 = self.conv2(g, x1)
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        
        # print(x2.shape)
        
        x2 = x2.view(self.time_step, 1140, -1)
        
        # print(x2.shape)
        
        # 执行LSTM操作
        xx, (h_n, c_n) = self.lstm1(x2)
        h_n = h_n.view(1140, -1)
        # print(h_n.shape)
        x3  = torch.cat((x[-1140:], h_n), dim=1)
        # print(x3.shape)
        x3 = self.bn3(x3)
        x3 = self.activation(x3)
        # x3 = F.relu(x3 + self.shortcut3(x2))
        
        # print(x3.shape)
        
        # 执行输出层操作
        active = self.active_layer1(x3)
        consume = self.consume_layer1(x3)
        
        active = self.activation(active)
        consume = self.activation(consume)
        
        active = self.active_layer2(active)
        consume = self.consume_layer2(consume)
        
        active = self.activation(active)
        consume = self.activation(consume)
        
        active = self.active_layer3(active)
        consume = self.consume_layer3(consume)
        
        # print(active.shape)
        # print(consume.shape)
        
        return active, consume

class CNN(nn.Module):
    def __init__(self, time_step):
        super(CNN, self).__init__()
        
        # Embedding Dim
        embed_dim = 10
                
        # NodeEmbedding层
        self.node_embedding = nn.Embedding(1140, embed_dim)
        
        # CNN层
        self.conv1 = nn.Conv1d(in_channels=43 + embed_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # BN层
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.bn4 = nn.BatchNorm1d(num_features=512)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.bn6 = nn.BatchNorm1d(num_features=256)
        
        # 输出层
        self.active_layer1 = nn.Linear(256*3, 256*3)
        self.active_layer2 = nn.Linear(256*3, 256*3)
        self.active_layer3 = nn.Linear(256*3, 1)
        self.consume_layer1 = nn.Linear(256*3, 256*3)
        self.consume_layer2 = nn.Linear(256*3, 256*3)
        self.consume_layer3 = nn.Linear(256*3, 1)
        
    def forward(self, data):
        
        # 获取数据
        x = data[:, :, 1:-2]
        node_id = data[:, :, 0]
        
        # 转换数据类型
        x = x.to(torch.float32)
        node_id = node_id.to(torch.long)
        
        if self.training:
            # 给x添加噪声
            x = x + (torch.randn(x.size()) * 0.1).to(device='cuda:0')
            # 对x进行随机缩放，缩放比例为0.8~1.2
            x = x * (torch.rand(x.size()) * 1 + 0.5).to(device='cuda:0')
        
        # 执行NodeEmbedding操作
        node_id = self.node_embedding(node_id)
        
        # 将Embedding的结果作为输入特征的一部分
        x = torch.cat((x, node_id), 2)
        
        # 交换维度
        x = x.permute(1, 2, 0)
        
        # print(x.shape)
        
        # 执行CNN操作
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = nn.Dropout(p=0.1)(x1)
        
        # print(x1.shape)
        
        # x1 = nn.MaxPool1d(kernel_size=2, stride=2)(x1)
        
        # print(x1.shape)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = nn.Dropout(p=0.1)(x2)
        
        # print(x2.shape)
        
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = nn.Dropout(p=0.1)(x3)
        
        # print(x3.shape)
        
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = nn.Dropout(p=0.1)(x4)
        
        # print(x4.shape)
        
        x4 = nn.MaxPool1d(kernel_size=2, stride=2)(x4)
        
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = F.relu(x5)
        x5 = nn.Dropout(p=0.1)(x5)
        
        # print(x5.shape)
        
        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = F.relu(x6)
        x6 = nn.Dropout(p=0.1)(x6)
        
        # print(x6.shape)
        
        # Flatten
        x = nn.Flatten()(x6)
        
        # print(x.shape)
        
        # 执行输出层操作
        active = self.active_layer1(x)
        consume = self.consume_layer1(x)
        
        active = F.relu(active)
        consume = F.relu(consume)
        
        active = self.active_layer2(active)
        consume = self.consume_layer2(consume)
        
        active = F.relu(active)
        consume = F.relu(consume)
        
        active = self.active_layer3(active)
        consume = self.consume_layer3(consume)
        
        return active, consume