import math
import pickle
import random
import time

import numpy as np
import torch_geometric

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader

from preprocess.sample_generator import kpi_titile, kpi_list
from visualization import t_sne
from util import *
from dataset import *
from model import *
import argparse



def train(model, train_x, train_y, batch_size, num_epochs, device):



    # Create the dataset and data loader
    train_dataset = SiameseDataset(train_x, train_y, device)

    # Create a data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(device)

    # Initialize the model, loss function, and optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletLoss(margin=1.0)



    # Training the model
    model.train()
    for epoch in range(num_epochs):
        accu_loss = 0
        iteration = 0
        for graph_batch11, graph_batch12, graph_batch13, graph_batch14, graph_batch21, graph_batch22, graph_batch23, graph_batch24, graph_batch31, graph_batch32, graph_batch33, graph_batch34 in train_loader:
            # graph1 = torch.tensor(graph1)
            # graph2 = torch.tensor(graph2)
            # print(graph1.x)
            # batch1 = Batch.from_data_list([graph1[i] for i in range(batch_size)])
            # batch2 = Batch.from_data_list([graph2[i] for i in range(batch_size)])
            # print(graph1.edge_index)
            optimizer.zero_grad()
            # print(batch1.x)
            # out, intermediate_output = model(data)
            out1 = model(graph_batch11, graph_batch12, graph_batch13, graph_batch14)
            out2 = model(graph_batch21, graph_batch22, graph_batch23, graph_batch24)
            out3 = model(graph_batch31, graph_batch32, graph_batch33, graph_batch34)
            loss = criterion(out1, out2, out3)
            loss.backward()
            optimizer.step()
            accu_loss += loss.item()
            iteration += 1
            # if iteration % 10 == 0:
            #     print(f'Iter {iteration}, Loss: {accu_loss/iteration:.4f}')
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {accu_loss / iteration:.4f}')

    return model


def test(model, train_sample_x, train_sample_y, test_sample_x, test_sample_y):

    model.eval()
    train_embeddings = []
    train_labels = []
    train_sample_embeddings = []
    train_sample_labels = []

    with torch.no_grad():
        for i in range(len(train_sample_x)):
            sample_pairs = train_sample_x[i]
            sample_graph1 = []
            sample_graph2 = []
            for j in range(len(sample_pairs)):
                sample_graph1.append(sample_pairs[j][0])
                sample_graph2.append(sample_pairs[j][1])
            batch1 = Batch.from_data_list(sample_graph1)
            batch2 = Batch.from_data_list(sample_graph2)
            out = model(batch1, batch2)
            train_embeddings.extend(out.cpu().numpy().tolist())
            train_labels.extend([train_sample_y[i]] * len(sample_pairs))
            train_sample_embeddings.append(out.cpu().numpy().tolist())
            train_sample_labels.append(train_sample_y[i])


    test_embeddings = []
    test_labels = []
    test_sample_embeddings = []
    test_sample_labels = []
    with torch.no_grad():
        for i in range(len(test_sample_x)):
            sample_pairs = test_sample_x[i]
            sample_graph1 = []
            sample_graph2 = []
            for j in range(len(sample_pairs)):
                sample_graph1.append(sample_pairs[j][0])
                sample_graph2.append(sample_pairs[j][1])
            batch1 = Batch.from_data_list(sample_graph1)
            batch2 = Batch.from_data_list(sample_graph2)
            out = model(batch1, batch2)
            test_embeddings.extend(out.cpu().numpy().tolist())
            test_labels.extend([test_sample_y[i]] * len(sample_pairs))
            test_sample_embeddings.append(out.cpu().numpy().tolist())
            test_sample_labels.append(test_sample_y[i])

    pred = get_prediction(test_sample_embeddings, train_sample_embeddings, train_labels)
    # print(pred)
    # print([e.item() for e in test_sample_labels])
    printresult(test_sample_labels, pred)

    pred = aggregate_train_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                          agg_method='median')
    # print(pred)
    # print([e.item() for e in test_sample_labels])
    printresult(test_sample_labels, pred)

    pred = aggregate_test_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                         agg_method='median')
    # print(pred)
    # print([e.item() for e in test_sample_labels])
    printresult(test_sample_labels, pred)

    pred = aggregate_train_test_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                               agg_method='median')
    # print(pred)
    # print([e.item() for e in test_sample_labels])
    printresult(test_sample_labels, pred)

    pred = election_based_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                         agg_method='median')
    # print(pred)
    # print([e.item() for e in test_sample_labels])
    printresult(test_sample_labels, pred)

    # t_sne(np.array(intermediate_outputs), np.array(y_true))



def interpret(model, test_sample_x, test_sample_y, test_replace_samples):
    model.eval()

    test_embeddings = []
    test_labels = []
    test_sample_embeddings = []
    test_sample_labels = []
    with torch.no_grad():
        for i in range(len(test_sample_x)):
            sample_pairs = test_sample_x[i]
            sample_graph1 = []
            sample_graph2 = []
            for j in range(len(sample_pairs)):
                sample_graph1.append(sample_pairs[j][0])
                sample_graph2.append(sample_pairs[j][1])
            batch1 = Batch.from_data_list(sample_graph1)
            batch2 = Batch.from_data_list(sample_graph2)
            out = model(batch1, batch2)
            test_embeddings.extend(out.cpu().numpy().tolist())
            test_labels.extend([test_sample_y[i]] * len(sample_pairs))
            test_sample_embeddings.append(out.cpu().numpy().tolist())
            test_sample_labels.append(test_sample_y[i])

    aggregate_test_sample_embeddings = []
    aggregate_test_replace_samples = []
    for i in range(len(test_sample_embeddings)):
        aggregare_ret = compute_medoid(test_sample_embeddings[i])
        aggregate_test_sample_embeddings.append(aggregare_ret[0])
        aggregate_test_replace_samples.append(test_replace_samples[i][aggregare_ret[1]])

    #############################
    aggregate_test_replace_sample_embeddings = []
    with torch.no_grad():
        for i in range(len(aggregate_test_replace_samples)):
            replace_samples = aggregate_test_replace_samples[i]

            sample_graph1 = []
            sample_graph2 = []
            for k in range(len(replace_samples)):
                sample_graph1.append(replace_samples[k][0])
                sample_graph2.append(replace_samples[k][1])
            batch1 = Batch.from_data_list(sample_graph1)
            batch2 = Batch.from_data_list(sample_graph2)
            out = model(batch1, batch2)
            aggregate_test_replace_sample_embeddings.append(out.cpu().numpy().tolist())

    dissimilarity = []
    for i in range(len(aggregate_test_replace_sample_embeddings)):
        replace_embedding = aggregate_test_replace_sample_embeddings[i]
        dist = []
        for k in range(len(replace_embedding)):
            dist.append(eucli_dist(replace_embedding[k], aggregate_test_sample_embeddings[i]))
        dissimilarity.append(dist)

    return dissimilarity, test_sample_y

def eval_interpret(dist, test_sample_y):
    y = test_sample_y
    dist_grouped = [[] for _ in range(9)]
    for i in range(len(y)):
        dist_grouped[y[i]].append(dist[i])

    # groundTruth = ['normF', 'int', 'grow', 'used', 'free', 'inodes', 'size', 'buff', 'writ', 'wai', 'blk']
    # groundTruth = ['grow', 'size', 'wai']
    # groundTruth = [['grow', 'buff', 'normF', 'free'],
    #                ['normF', 'free', 'buff'],
    #                ['size', 'normF', 'free', 'buff', 'used'],
    #                ['normF', 'free', 'buff'],
    #                ['normF', 'free', 'inodes'],
    #                ['Locks', 'buff', 'writ'],
    #                ['size', 'normF', 'free', 'buff'],
    #                ['normF', 'free', 'buff', 'used'],
    #                ['size', 'normF', 'free', 'buff']]
    groundTruth = [['grow', 'buff', 'normF'],
                   ['normF', 'free', 'buff'],
                   ['size', 'normF', 'buff'],
                   ['normF', 'free', 'buff'],
                   ['normF', 'free', 'inodes'],
                   ['Locks', 'buff', 'writ'],
                   ['size', 'normF', 'buff'],
                   ['normF', 'free', 'buff'],
                   ['size', 'normF', 'buff']]
    hit_100_average = 0
    hit_150_average = 0
    for i in range(9):
        anomaly_kpis = []
        for e in dist_grouped[i]:
            # 按降序排序
            sorted_indices_desc = np.argsort(e)[::-1]
            anomaly_kpis.append([kpi_titile[kpi_list[i]] for i in sorted_indices_desc])

        hit_100 = 0
        hit_150 = 0
        for e in anomaly_kpis:
            for ee in e[:len(groundTruth[i])]:
                if ee in groundTruth[i]:
                    hit_100 += 1
            for ee in e[:int(len(groundTruth[i]) * 1.5) + 1]:
                if ee in groundTruth[i]:
                    hit_150 += 1
        hit_100 = hit_100 / (len(anomaly_kpis) * len(groundTruth[i]))
        hit_150 = hit_150 / (len(anomaly_kpis) * len(groundTruth[i]))
        hit_100_average += hit_100
        hit_150_average += hit_150
    hit_100_average = hit_100_average / 9
    hit_150_average = hit_150_average / 9
    print(hit_100_average)
    print(hit_150_average)



if __name__ == '__main__':
    # Setting a random seed
    seed = 17
    torch.manual_seed(seed)
    torch_geometric.seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = 32
    num_epochs = 20

    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('mtcn_c', type=int,
                        help='the number of channels of the convolution kernel of mtcn')
    parser.add_argument('p', type=int,
                        help='the number of lower layers of graph convolution')

    parser.add_argument('q', type=int,
                        help='the number of upper layers of graph convolution')

    parser.add_argument('latent_size', type=int,
                        help='the dimension of the discrepency representation')
    args = parser.parse_args()


    dataset_path = r'dataset/preprocessed/train_10_05_dataset_1_full.pickle'
    # Check if the GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # graph_pairs, y = load_pair_graph_dataset('train_10_05_dataset.pickle', device)

    train_x, train_y, train_sample_x, train_sample_y = load_train_dataset(dataset_path, device)

    model = SDNEncoder(cnn_hidden_channels=args.mtcn_c, p=args.p, q=args.q,graph_hidden_channels=8, discrepancy_hidden_channels=8, out_channels=args.latent_size,
                       num_nodes=97,
                       window_size=10)
    model = model.to(device)

    train(model, train_x, train_y, batch_size, num_epochs=num_epochs, device=device)

    test_sample_x, test_sample_y = load_test_dataset('dataset/without_trend/test_10_05_dataset_1_full.pickle', device)

    test(model, train_sample_x, train_sample_y, test_sample_x, test_sample_y)

    test_interpret_samples = load_interpret_dataset('dataset/without_trend/test_interpret_10_05_dataset_1.pickle',
                                                    device)

    inter_tem = interpret(model, test_sample_x, test_sample_y, test_interpret_samples)
    eval_interpret(*inter_tem)



    torch.save(model.state_dict(),
               f'trained_model/train_10_05_model_{time.strftime("%d-%H_%M_%S", time.localtime())}.pth')




