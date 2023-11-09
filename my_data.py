import math
import torch

class DataIterator(object):
    def __init__(self, x_data, x_mask_data, x_edge_data, args):
        # 初始化数据
        self.x_data, self.x_mask_data, self.x_edge_data = x_data, x_mask_data, x_edge_data

        # 将日期和特征分开
        self.x_date, self.x_feature, self.x_tags = self.x_data[:,:,0], self.x_data[:,:,1:-2], x_data[:,:,-2:]

        self.args = args

        # 计算批次数量
        self.batch_count = math.ceil(len(x_data)/args.batch_size)

    def get_batch(self, index):
        # 初始化批次数据
        x_date = []
        x_feature = []
        x_mask_data = []
        x_edge_data = []
        x_tags = []

        # 获取批次数据
        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.x_data))):

            x_date.append(self.x_date[i])
            x_feature.append(self.x_feature[i].float())
            x_mask_data.append(self.x_mask_data[i])
            x_edge_data.append(self.x_edge_data[i])
            x_tags.append(self.x_tags[i].float())

        # 将数据转换为张量
        x_date = torch.stack(x_date).to(self.args.device)
        x_feature = torch.FloatTensor(torch.stack(x_feature)).to(self.args.device)
        x_mask_data = torch.stack(x_mask_data).to(self.args.device)
        x_edge_data = torch.stack(x_edge_data).to(self.args.device)
        x_tags = torch.stack(x_tags).to(self.args.device)

        return x_date, x_feature, x_mask_data, x_edge_data, x_tags