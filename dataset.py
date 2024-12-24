import pickle

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SiameseDataset(Dataset):
    def __init__(self, graph_pairs, labels, device):
        super(SiameseDataset, self).__init__()
        self.graph_pairs = graph_pairs
        self.labels = np.array(labels)
        self.device = device

    def __len__(self):
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        graph_pair1 = self.graph_pairs[idx]

        positive_idx = np.random.choice(np.where(self.labels == self.labels[idx])[0])
        negative_idx = np.random.choice(np.where(self.labels != self.labels[idx])[0])

        positive = self.graph_pairs[positive_idx]
        negative = self.graph_pairs[negative_idx]

        # graph_pair1 = torch.tensor(graph_pair1)
        # graph_pair2 = torch.tensor(graph_pair2)

        return graph_pair1[0], graph_pair1[1], graph_pair1[2], graph_pair1[3], positive[0], positive[1], positive[2], positive[3], negative[0], negative[1], negative[2], negative[3]


def load_pair_graph_dataset(path, device):
    x, y = pickle.load(open(path, 'rb'))
    graph_pairs = []
    for i in range(len(x)):
        graph_pairs.append((Data(x=torch.tensor(x[i][0][0], dtype=torch.float).to(device),
                                 edge_index=torch.tensor(x[i][0][1], dtype=torch.long)).to(device),
                            Data(x=torch.tensor(x[i][1][0], dtype=torch.float).to(device),
                                 edge_index=torch.tensor(x[i][1][1], dtype=torch.long).to(device)),
                            Data(x=None,
                                 edge_index=torch.tensor(x[i][0][2], dtype=torch.long).to(device)),
                            Data(x=None,
                                 edge_index=torch.tensor(x[i][1][2], dtype=torch.long).to(device))
                            ))
    # y = torch.tensor(y, dtype=torch.long).to(device)
    return graph_pairs, y


def load_train_dataset(path, device):
    x, y, sample_x, sample_y = pickle.load(open(path, 'rb'))
    graph_pairs = []
    for i in range(len(x)):
        graph_pairs.append((Data(x=torch.tensor(x[i][0][0], dtype=torch.float).to(device),
                                 edge_index=torch.tensor(x[i][0][1], dtype=torch.long)).to(device),
                            Data(x=torch.tensor(x[i][1][0], dtype=torch.float).to(device),
                                 edge_index=torch.tensor(x[i][1][1], dtype=torch.long).to(device))))
    # y = torch.tensor(y, dtype=torch.long).to(device)
    sample_pairs = []
    for i in range(len(sample_x)):
        t = []
        for j in range(len(sample_x[i])):
            t.append((Data(x=torch.tensor(sample_x[i][j][0][0], dtype=torch.float).to(device),
                           edge_index=torch.tensor(sample_x[i][j][0][1], dtype=torch.long)).to(device),
                      Data(x=torch.tensor(sample_x[i][j][1][0], dtype=torch.float).to(device),
                           edge_index=torch.tensor(sample_x[i][j][1][1], dtype=torch.long).to(device))))
        sample_pairs.append(t)
    # sample_y = torch.tensor(sample_y, dtype=torch.long).to(device)
    return graph_pairs, y, sample_pairs, sample_y


def load_test_dataset(path, device):
    sample_x, sample_y = pickle.load(open(path, 'rb'))
    sample_pairs = []
    for i in range(len(sample_x)):
        t = []
        for j in range(len(sample_x[i])):
            t.append((Data(x=torch.tensor(sample_x[i][j][0][0], dtype=torch.float).to(device),
                           edge_index=torch.tensor(sample_x[i][j][0][1], dtype=torch.long)).to(device),
                      Data(x=torch.tensor(sample_x[i][j][1][0], dtype=torch.float).to(device),
                           edge_index=torch.tensor(sample_x[i][j][1][1], dtype=torch.long).to(device))))
        sample_pairs.append(t)
    # sample_y = torch.tensor(sample_y, dtype=torch.long).to(device)
    return sample_pairs, sample_y

def load_interpret_dataset(path, device):
    interpret_samples, _ = pickle.load(open(path, 'rb'))
    samples = []
    for i in range(len(interpret_samples)):
        sample = interpret_samples[i]
        t = []
        for j in range(len(sample)):
            t1 = []
            for k in range(len(sample[j])):
                pair = sample[j][k]
                t1.append((Data(x=torch.tensor(pair[0][0], dtype=torch.float).to(device),
                               edge_index=torch.tensor(pair[0][1], dtype=torch.long)).to(device),
                          Data(x=torch.tensor(pair[1][0], dtype=torch.float).to(device),
                               edge_index=torch.tensor(pair[1][1], dtype=torch.long).to(device))))
            t.append(t1)
        samples.append(t)
    return samples


