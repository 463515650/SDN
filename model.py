import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class HierarchicalGCN(nn.Module):
    def __init__(self, group2nodes, num_nodes, node_dim):
        super(HierarchicalGCN, self).__init__()
        self.group2nodes = group2nodes
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.aggregate_feedforward = nn.ModuleList([nn.Linear(node_dim*len(nodes), node_dim) for group,nodes in group2nodes.items()])
        self.lowerGCN = GCNConv(node_dim, node_dim)
        self.upperGCN = GCNConv(node_dim, node_dim)

    def forward(self, x, A1, A2):
        x = self.lowerGCN(x, A1)
        group_x = []
        for group,nodes in self.group2nodes.items():
            node_x = []
            for n in nodes:
                node_x.append(x[n::self.num_nodes])
            node_x = torch.stack(node_x, dim=1)
            node_x = torch.flatten(node_x, start_dim=1)
            group_x.append(self.aggregate_feedforward[group](node_x))
        group_x = torch.stack(group_x, dim=1)
        group_x = group_x.reshape(-1, self.node_dim)
        group_x = self.upperGCN(group_x, A2)
        return group_x




class MultiscaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleCNN, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.tconv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.tconv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.tconv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    # def forward(self, x):
    #     x_conv3 = self.conv3(x)
    #     out1 = torch.max(x_conv3, dim=2)
    #     x_conv5 = self.conv5(x)
    #     out2 = torch.max(x_conv5, dim=2)
    #     x_conv7 = self.conv7(x)
    #     out3 = torch.max(x_conv7, dim=2)
    #     return torch.cat([out1, out2, out3], dim=1)

    def forward(self, x):
        x_conv = x[:,0,:].unsqueeze(1)
        x_conv3 = self.conv3(x_conv)
        out1 = torch.max(x_conv3, dim=2)[0]
        x_conv5 = self.conv5(x_conv)
        out2 = torch.max(x_conv5, dim=2)[0]
        x_conv7 = self.conv7(x_conv)
        out3 = torch.max(x_conv7, dim=2)[0]
        x_tconv3 = self.tconv3(x[:,1,:-2].unsqueeze(1))
        out4 = torch.max(x_tconv3, dim=2)[0]
        x_tconv5 = self.tconv5(x[:, 2, :-4].unsqueeze(1))
        out5 = torch.max(x_tconv5, dim=2)[0]
        x_tconv7 = self.tconv7(x[:, 3, :-6].unsqueeze(1))
        out6 = torch.max(x_tconv7, dim=2)[0]
        return torch.cat([out1, out2, out3, out4, out5, out6], dim=1)


# 定义图卷积网络模型
class SDNEncoder(nn.Module):
    def __init__(self, cnn_hidden_channels, graph_hidden_channels, discrepancy_hidden_channels, out_channels, num_nodes, window_size, group2nodes):
        super(SDNEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.graph_hidden_channels = graph_hidden_channels
        self.group2nodes = group2nodes
        self.num_groups = len(self.group2nodes)
        # self.positional_embeddings = nn.Parameter(torch.randn(num_nodes, in_channels))
        self.multiCNN = nn.ModuleList([MultiscaleCNN(1, cnn_hidden_channels) for _ in range(num_nodes)])

        self.feedforward = nn.ModuleList([nn.Linear(6 * cnn_hidden_channels, graph_hidden_channels) for _ in range(num_nodes)])

        # self.conv0 = GCNConv(6 * window_size, 6 * window_size)
        # self.conv1 = GCNConv(6 * window_size, graph_hidden_channels)
        # self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)

        self.hgcn = HierarchicalGCN(group2nodes, num_nodes, graph_hidden_channels)

        self.discrepancy_fc = nn.ModuleList(
            [nn.Linear(2 * graph_hidden_channels, discrepancy_hidden_channels) for _ in range(self.num_groups)])

        self.fc = nn.Linear(self.num_groups * discrepancy_hidden_channels, out_channels)

    def forward(self, graph1, graph2, group_graph1, group_graph2):
        x1, edge_index1, batch1 = graph1.x, graph1.edge_index, graph1.batch
        x2, edge_index2, batch2 = graph2.x, graph2.edge_index, graph2.batch
        _, group_edge_index1, _ = group_graph1.x, group_graph1.edge_index, group_graph1.batch
        _, group_edge_index2, _ = group_graph2.x, group_graph2.edge_index, group_graph2.batch

        batch_size = x1.size()[0]//self.num_nodes
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, batch = self.positional_embeddings, data.edge_index, data.batch
        # x = x.repeat(len(data.x)//self.num_nodes, 1)
        cnn_outputs1 = []

        for n in range(self.num_nodes):
            cnn_output = self.multiCNN[n](x1[n::self.num_nodes])
            cnn_outputs1.append(cnn_output)

        cnn_outputs2 = []

        for n in range(self.num_nodes):
            cnn_output = self.multiCNN[n](x2[n::self.num_nodes])
            cnn_outputs2.append(cnn_output)

        x1 = torch.stack(cnn_outputs1)
        x2 = torch.stack(cnn_outputs2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)

        feedforward_outputs1 = []
        for n in range(self.num_nodes):
            feedforward_output = self.feedforward[n](x1[n])
            feedforward_outputs1.append(feedforward_output)

        feedforward_outputs2 = []
        for n in range(self.num_nodes):
            feedforward_output = self.feedforward[n](x2[n])
            feedforward_outputs2.append(feedforward_output)

        x1 = torch.stack(feedforward_outputs1)
        x2 = torch.stack(feedforward_outputs2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)

        x1 = torch.transpose(x1, 0, 1)
        x1 = x1.reshape((-1, self.graph_hidden_channels))
        x1 = self.hgcn(x1, edge_index1, group_edge_index1)
        x1 = F.relu(x1)

        x2 = torch.transpose(x2, 0, 1)
        x2 = x2.reshape((-1, self.graph_hidden_channels))
        x2 = self.hgcn(x2, edge_index2, group_edge_index2)
        x2 = F.relu(x2)

        discrepancy_outputs = []
        for n in range(self.num_groups):
            discrepancy_output = self.discrepancy_fc[n](torch.cat([x1[n::self.num_groups], x2[n::self.num_groups]], dim=1))
            discrepancy_outputs.append(discrepancy_output)
        discrepancy_outputs = torch.stack(discrepancy_outputs)
        discrepancy_outputs = F.relu(discrepancy_outputs)

        out = self.fc(torch.transpose(discrepancy_outputs, 0, 1).reshape((batch_size, -1)))

        return out


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
