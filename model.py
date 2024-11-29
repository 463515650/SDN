import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class MultiscaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleCNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    # def forward(self, x):
    #     x_conv3 = self.conv3(x)
    #     out1 = torch.max(x_conv3, dim=2)
    #     x_conv5 = self.conv5(x)
    #     out2 = torch.max(x_conv5, dim=2)
    #     x_conv7 = self.conv7(x)
    #     out3 = torch.max(x_conv7, dim=2)
    #     return torch.cat([out1, out2, out3], dim=1)

    def forward(self, x):
        x_conv3 = self.conv3(x)
        x_conv5 = self.conv5(x)
        x_conv7 = self.conv7(x)
        return torch.cat([x_conv3, x_conv5, x_conv7], dim=1)


# 定义图卷积网络模型
class SDNEncoder(nn.Module):
    def __init__(self, graph_hidden_channels, discrepancy_hidden_channels, out_channels, num_nodes, window_size):
        super(SDNEncoder, self).__init__()
        self.num_nodes = num_nodes
        # self.positional_embeddings = nn.Parameter(torch.randn(num_nodes, in_channels))
        self.multiCNN = nn.ModuleList([MultiscaleCNN(1, 2) for _ in range(num_nodes)])

        self.feedforward = nn.ModuleList([nn.Linear(6 * window_size, graph_hidden_channels) for _ in range(num_nodes)])

        # self.conv0 = GCNConv(6 * window_size, 6 * window_size)
        # self.conv1 = GCNConv(6 * window_size, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)

        self.discrepancy_fc = nn.ModuleList(
            [nn.Linear(2 * graph_hidden_channels, discrepancy_hidden_channels) for _ in range(num_nodes)])

        self.fc = nn.Linear(num_nodes * discrepancy_hidden_channels, out_channels)

    def forward(self, graph1, graph2):
        x1, edge_index1, batch1 = graph1.x, graph1.edge_index, graph1.batch
        x2, edge_index2, batch2 = graph2.x, graph2.edge_index, graph2.batch

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, batch = self.positional_embeddings, data.edge_index, data.batch
        # x = x.repeat(len(data.x)//self.num_nodes, 1)
        cnn_outputs1 = []

        for i in range(x1.size(0)):
            cnn_output = self.multiCNN[i % self.num_nodes](x1[i].unsqueeze(0).unsqueeze(1))
            cnn_outputs1.append(cnn_output.view(-1))

        cnn_outputs2 = []
        for i in range(x2.size(0)):
            cnn_output = self.multiCNN[i % self.num_nodes](x2[i].unsqueeze(0).unsqueeze(1))
            cnn_outputs2.append(cnn_output.view(-1))

        x1 = torch.stack(cnn_outputs1)
        x2 = torch.stack(cnn_outputs2)

        x1 = F.relu(x1)
        x2 = F.relu(x2)


        feedforward_outputs1 = []
        for i in range(x1.size(0)):
            feedforward_output = self.feedforward[i % self.num_nodes](x1[i].unsqueeze(0))
            feedforward_outputs1.append(feedforward_output.view(-1))

        feedforward_outputs2 = []
        for i in range(x2.size(0)):
            feedforward_output = self.feedforward[i % self.num_nodes](x2[i].unsqueeze(0))
            feedforward_outputs2.append(feedforward_output.view(-1))

        x1 = torch.stack(feedforward_outputs1)  # 堆叠输出
        x2 = torch.stack(feedforward_outputs2)

        x1 = F.relu(x1)
        x2 = F.relu(x2)

        # x1 = self.conv0(x1, edge_index1)
        # x1 = F.relu(x1)
        # x1 = self.conv1(x1, edge_index1)
        # x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)

        # x2 = self.conv0(x2, edge_index2)
        # x2 = F.relu(x2)
        # x2 = self.conv1(x2, edge_index2)
        # x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)

        discrepancy_outputs = []
        for i in range(x1.size(0)):
            discrepancy_output = self.discrepancy_fc[i % self.num_nodes](
                torch.cat([x1[i], x2[i]]).view(-1).unsqueeze(0))
            discrepancy_outputs.append(discrepancy_output.view(-1))
        discrepancy_units = []
        for i in range(self.num_nodes):
            discrepancy_units.append(torch.stack(discrepancy_outputs[i::self.num_nodes]))
        discrepancy_units = torch.cat(discrepancy_units, 1)
        discrepancy_units = F.relu(discrepancy_units)

        out = self.fc(discrepancy_units)

        return out

        # # 使用全局池化层
        # x = global_mean_pool(x, batch)
        # intermediate_output = x
        # x = self.fc(x)
        # return F.log_softmax(out, dim=1)


class CSDN(nn.Module):
    def __init__(self, graph_hidden_channels, discrepancy_hidden_channels, out_channels, num_nodes, window_size):
        super(CSDN, self).__init__()
        self.SDN = SDNEncoder(graph_hidden_channels, discrepancy_hidden_channels, out_channels, num_nodes, window_size)

    def forward(self, graph_pair1, graph_pair2):
        encode1 = self.SDN(*graph_pair1)
        encode2 = self.SDN(*graph_pair2)
        return encode1, encode2


# Triplet损失类
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive)
        negative_distance = F.pairwise_distance(anchor, negative)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算欧几里得距离
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # 计算损失
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return torch.mean(loss)
