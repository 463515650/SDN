import argparse
import copy
import pickle
import random

import pandas as pd
from matplotlib import style, pyplot as plt
from pgmpy.estimators import PC
from scipy.stats import norm, stats, linregress

import numpy as np
import networkx as nx

from preprocess.dataset import multidataset,dataset

# kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
kpi_titile = ["Time","usr","sys","idl","wai","stl","read","writ","recv","send","in","out","used","free","buff","cach",
              "int","csw","run","blk","new","1m","5m","15m","used","free","343","344","416","read","writ","#aio",
              "files","inodes","msg","sem","shm","pos","lck","rea","wri","raw","tot","tcp","udp","raw","frg","lis",
              "act","syn","tim","clo","lis","act","dgm","str","lis","act","majpf","minpf","alloc","free","steal",
              "scanK","scanD","pgoru","astll","d32F","d32H","normF","normH","Conn","%Con","Act","LongQ","LongX",
              "Idl","LIdl","LWait","SQLs1","SQLs3","SQLs5","Xact1","Xact3","Locks","shared_buffers","work_mem",
              "bgwriter_delay","max_connections","autovacuum_work_mem","temp_buffers","autovacuum_max_workers",
              "maintenance_work_mem","checkpoint_timeout","max_wal_size","checkpoint_completion_target",
              "wal_keep_segments","wal_segment_size","clean","back","alloc","heapr","heaph","ratio","size","grow",
              "insert","update","delete","comm","roll"]
group_to_kpi = {1:[_ for _ in range(1,6)], 2:[_ for _ in range(6,8)], 3:[_ for _ in range(8,9)], 4:[_ for _ in range(9,11)],
            5:[_ for _ in range(12,16)], 6:[_ for _ in range(16,18)], 7:[_ for _ in range(18,21)],
            8:[_ for _ in range(21,24)], 9:[_ for _ in range(24,26)], 10:[_ for _ in range(26,29)],
            11:[_ for _ in range(29,31)], 12:[_ for _ in range(31,32)], 13:[_ for _ in range(32,34)],
            14:[_ for _ in range(34,37)], 15:[_ for _ in range(37,41)], 16:[_ for _ in range(41,42)],
            17:[_ for _ in range(42,47)], 18:[_ for _ in range(47,52)], 19:[_ for _ in range(52,54)],
            20:[_ for _ in range(54,58)], 21:[_ for _ in range(58,62)], 22:[_ for _ in range(62,67)],
            23:[_ for _ in range(67,71)], 24:[_ for _ in range(71,84)], 25:[_ for _ in range(84,85)],
            26:[_ for _ in range(98,111)]}
# kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]
# kpis selected
kpi_list = [i for i in range(1, 111)]
for e in range(85,98):
    kpi_list.remove(e)

kpi_to_group = {}

for group,kpis in group_to_kpi.items():
    for kpi in kpis:
        if kpi >= 98:
            kpi = kpi - 13
        kpi_to_group[kpi - 1] = group - 1

def fisher_z_test(r, n):

    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_score = z / se

    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return z, z_score, p_value

def correlation_graph(data):
    kpi_length = data.shape[0]
    cor = np.corrcoef(np.array(data[:, kpi_list].tolist()), rowvar=False)
    cor[np.isnan(cor)] = 0
    test_result = {}
    for i in range(len(kpi_list)):
        for j in range(i+1, len(kpi_list)):
            test_result[(i, j)] = fisher_z_test(cor[i, j],kpi_length)

    return cor, test_result



def draw_correlation(data):
    cors = []
    for i in range(len(data)):
        cor = np.corrcoef(np.array(data[i][:, kpi_list].tolist()), rowvar=False)
        cors.append(cor)
    cors = np.array(cors)
    median = np.median(cors, axis=0)
    mad = np.median(np.absolute(cors - median), axis=0)

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(kpi_list)
    for i in range(len(mad)):
        for j in range(i + 1, len(mad[i])):
            if mad[i][j] <= 0.1 and median[i][j] >= 0.7:
                G.add_edge(kpi_list[i], kpi_list[j], weight=str(np.round(median[i][j],2)))

    labels = {n: kpi_titile[n] for n in kpi_list}
    edges_labels = nx.get_edge_attributes(G, 'weight')

    # 设置布局
    pos = nx.spring_layout(G, k=1.5)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels)
    plt.show()

def do_linear_regression(series, scale):
    ret = []
    time = [_ for _ in range(scale)]
    for i in range(0,len(series)-scale+1):
        y = series[i:i+scale]
        slope = linregress(time, y)[0]
        ret.append(slope)
    return ret


def get_correlation_graph(data, normalized_data, with_trend=True, scales=[3,5,7]):
    cor, test_result = correlation_graph(data)
    x = normalized_data[:, np.array(kpi_list) - 1].transpose().tolist()
    if with_trend:
        x_with_trend = []
        for i in range(len(kpi_list)):
            t = []
            t.append(x[i])
            for s in scales:
                t.append(do_linear_regression(x[i] + [0]*(s-1), s))
            x_with_trend.append(t)
        x = x_with_trend
    edge_index = [[], []]
    for key in test_result:
        value = test_result[key]
        if value[2] < 0.01:
            edge_index[0].append(key[0])
            edge_index[1].append(key[1])
            edge_index[0].append(key[1])
            edge_index[1].append(key[0])
    return x, edge_index, get_group_correlation_edges(edge_index)

def get_group_correlation_edges(kpi_edges):
    group_edges = [[],[]]
    v1 = kpi_edges[0]
    v2 = kpi_edges[1]
    group_edges_set = set()
    for i in range(len(v1)):
        p = v1[i]
        q = v2[i]
        group_edges_set.add((kpi_to_group[p], kpi_to_group[q]))

    for edge in group_edges_set:
        group_edges[0].append(edge[0])
        group_edges[1].append(edge[1])

    return group_edges


def generate_graph_dataset(data):
    x = []
    y = []
    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            x.append(get_correlation_graph(data.aligned_data[f][i], data.normalized_data[f][i]))
            y.append(f)
    for i in range(len(data.normal_data)):
        x.append(get_correlation_graph(data.normal_data[i], data.normalized_normal[i]))
        y.append(9)
    return x, y

def sample_random_pair(normal, anomaly, k):
    pairs = []
    selected = np.random.choice(len(normal), size=k, replace=False)
    normal_selected = []
    for i in selected:
        normal_selected.append(normal[i])
    for i in range(len(anomaly)):
        for j in range(len(normal_selected)):
            pairs.append((normal_selected[j], anomaly[i]))

    return pairs


def generate_graph_clustered_pairs_dataset(data):
    all_pairs_x = []
    y = []
    sample_pairs = [] # [ [(normal1, sample1),(normal2, sample1)],... ]
    sample_y = []
    normal_x = [[] for _ in range(len(data.datasets))]

    for i in range(len(data.clustered_normal_index)):
        for j in data.clustered_normal_index[i]:
            normal = data.aligned_normal[j]
            normalized_normal = data.normalized_normal[j]
            normal_x[i].append(get_correlation_graph(normal, normalized_normal))

    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            anomaly_x = get_correlation_graph(data.aligned_data[f][i], data.normalized_feature[f][i])
            pairs_x = []
            for j in normal_x[data.index[f][i]]:
                pairs_x.append((j, anomaly_x))
            all_pairs_x.extend(pairs_x)
            y.extend([f] * len(pairs_x))
            sample_pairs.append(pairs_x)
            sample_y.append(f)
    return all_pairs_x, y, sample_pairs, sample_y


def generate_graph_constrained_pairs_dataset(data, k):
    all_pairs_x = []
    y = []
    sample_pairs = [] # [ [(normal1, sample1),(normal2, sample1)],... ]
    sample_y = []
    normal_x = [[] for _ in range(len(data.datasets))]
    for i in range(len(data.aligned_normal)):
        normal_x[data.normal_index[i]].append(get_correlation_graph(data.aligned_normal[i], data.normalized_normal[i]))

    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            anomaly_x = [get_correlation_graph(data.aligned_data[f][i], data.normalized_feature[f][i])]
            pairs_x = sample_random_pair(normal_x[data.index[f][i]], anomaly_x, k)
            all_pairs_x.extend(pairs_x)
            y.extend([f] * len(pairs_x))
            sample_pairs.append(pairs_x)
            sample_y.append(f)
    return all_pairs_x, y, sample_pairs, sample_y


# deprecated
def generate_graph_random_pairs_dataset(data, k):
    normal_x = []
    anomaly_x = [[] for _ in range(9)]
    pairs_x = []
    y = []
    for i in range(len(data.aligned_normal)):
        normal_x.append(get_correlation_graph(data.aligned_normal[i], data.normalized_normal[i]))
    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            anomaly_x[f].append(get_correlation_graph(data.aligned_data[f][i], data.normalized_feature[f][i]))
    for f in range(9):
        pairs = sample_random_pair(normal_x, anomaly_x[f], k)
        pairs_x.extend(pairs)
        y.extend([f] * len(pairs))
    return pairs_x, y

# 暂时无用
def generate_dual_pairs_dataset(data, k):
    normal_x = []
    anomaly_x = [[] for _ in range(9)]
    pairs_x = []
    dual_pairs = []
    y = []
    for i in range(len(data.aligned_normal)):
        normal_x.append(get_correlation_graph(data.aligned_normal[i], data.normalized_normal[i]))
    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            anomaly_x[f].append(get_correlation_graph(data.aligned_data[f][i], data.normalized_feature[f][i]))
    for f in range(9):
        pairs = sample_random_pair(normal_x, anomaly_x[f], k)
        pairs_x.append(pairs)
    for f in range(9):
        for i in range(len(pairs_x[f])):
            for j in range(i+1, len(pairs_x[f])):
                dual_pairs.append((pairs_x[f][i], pairs_x[f][j]))
                y.append(0)
    for f in range(9):
        for i in range(len(pairs_x[f])):
            for f1 in range(f+1, 9):
                for j in range(len(pairs_x[f1])):
                    dual_pairs.append((pairs_x[f][i], pairs_x[f1][j]))
                    y.append(1)

    return dual_pairs, y

def generate_train_test_dataset(dataset_list, ratio, k):
    train_list = []
    test_list = []
    for dataset in dataset_list:
        #dataset.clip(10, 100)
        dataset.cluster_normal(k, kpi_list)
        train, test = dataset.split(ratio)
        train_list.append(train)
        test_list.append(test)
    train_dataset = multidataset(train_list)
    test_dataset = multidataset(test_list)
    train_dataset.load()
    test_dataset.load()
    multidataset.joint_normalize(train_dataset, test_dataset)

    return train_dataset, test_dataset

def generate_test_graph_pairs(test_data, k):

    sample_pairs = []  # [ [(normal1, sample1),(normal2, sample1)],... ]
    y = []
    normal_x = [[] for _ in range(len(test_data.datasets))]
    for i in range(len(test_data.aligned_normal)):
        normal_x[test_data.normal_index[i]].append(get_correlation_graph(test_data.aligned_normal[i], test_data.normalized_normal[i]))

    for f in range(9):
        for i in range(len(test_data.aligned_data[f])):
            anomaly_x = [get_correlation_graph(test_data.aligned_data[f][i], test_data.normalized_feature[f][i])]
            pairs_x = sample_random_pair(normal_x[test_data.index[f][i]], anomaly_x, k)
            sample_pairs.append(pairs_x)
            y.append(f)

    return sample_pairs, y

def generate_test_clustered_pairs(data):
    sample_pairs = [] # [ [(normal1, sample1),(normal2, sample1)],... ]
    y = []
    normal_x = [[] for _ in range(len(data.datasets))]

    for i in range(len(data.clustered_normal_index)):
        for j in data.clustered_normal_index[i]:
            normal = data.aligned_normal[j]
            normalized_normal = data.normalized_normal[j]
            normal_x[i].append(get_correlation_graph(normal, normalized_normal))

    for f in range(9):
        for i in range(len(data.aligned_data[f])):
            anomaly_x = get_correlation_graph(data.aligned_data[f][i], data.normalized_feature[f][i])
            pairs_x = []
            for j in normal_x[data.index[f][i]]:
                pairs_x.append((j, anomaly_x))
            sample_pairs.append(pairs_x)
            y.append(f)
    return sample_pairs, y


def replace_kpi(pair, kpi):
    new_pair = copy.deepcopy(pair)
    normal_graph = new_pair[0]
    anomaly_graph = new_pair[1]
    kpi_index = kpi_list.index(kpi)
    anomaly_graph[0][kpi_index] = normal_graph[0][kpi_index]
    return new_pair



def generate_test_interpret_dataset(sample_pairs):
    interpret_samples = [] # [ [ [(normal,anomaly_kpi1)] ] ]

    for sample in sample_pairs:
        replace_sample = []
        for pair in sample:
            t = []
            for k in kpi_list:
                t.append(replace_kpi(pair, k))
            replace_sample.append(t)
        interpret_samples.append(replace_sample)

    return interpret_samples



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate samples.')
    parser.add_argument('-k', type=int, default=5,
                        help='the number of selected normal samples, eg 5')
    parser.add_argument('-test_ratio', type=float, default=0.5,
                        help='the ratio of test samples, eg 0.5')

    parser.add_argument('-sample_strategy', type=str, default='clustered',
                        help='sampling strategy, eg random, clustered')

    args = parser.parse_args()


    seed = 17
    np.random.seed(seed)
    random.seed(seed)

    datasets_path = [r'dataset\32_128.data',
                     r'dataset\32_256.data',
                     r'dataset\64_128.data',
                     r'dataset\64_256.data']
    datasets = []

    for path in datasets_path:
        datasets.append(pickle.load(open(path, 'rb')))

    k = args.k

    train, test = generate_train_test_dataset(datasets, args.test_ratio, k)

    ######################################################
    ISQUAD_dataset = False
    DBSherlock_dataset = False
    if ISQUAD_dataset or DBSherlock_dataset:
        train_x = []
        train_y = []
        train_cluster_index = []
        test_x = []
        test_y = []
        test_cluster_index = []
        for i in range(len(train.aligned_data)):
            t = train.aligned_data[i]
            train_x.extend(t)
            train_y.extend([i] * len(t))
            train_cluster_index.extend(train.index[i])
        for i in range(len(test.aligned_data)):
            t = test.aligned_data[i]
            test_x.extend(t)
            test_y.extend([i] * len(t))
            test_cluster_index.extend(test.index[i])
        normal_data = [[] for _ in range(len(train.datasets))]
        for i in range(len(train.aligned_normal)):
            normal_data[train.normal_index[i]].append(train.aligned_normal[i])
        if ISQUAD_dataset:
            pickle.dump((train_x, train_y, test_x, test_y), open('isquad_train_test_10_05_dataset.pickle', 'wb'))
        if DBSherlock_dataset:
            pickle.dump((train_x, train_y, train_cluster_index, test_x, test_y, test_cluster_index, normal_data), open('dbsherlock_train_test_10_05_dataset.pickle', 'wb'))
    ########################################################


    strategy = args.sample_strategy  # 0: random, 1: cluster-based

    if strategy == 'random':
        x, y, sample_x, sample_y = generate_graph_constrained_pairs_dataset(train, k)
        print('the number of training samples：' + str(len(y)))
        pickle.dump((x, y, sample_x, sample_y), open('dataset/preprocessed/train_10_05_dataset_0.pickle', 'wb'))

        sample_pairs, test_y = generate_test_graph_pairs(test, k)
        print('the number of testing samples：' + str(len(test_y)))
        pickle.dump((sample_pairs, test_y), open('dataset/preprocessed/test_10_05_dataset_0.pickle', 'wb'))

    else:
        x, y, sample_x, sample_y = generate_graph_clustered_pairs_dataset(train)
        print('the number of training samples' + str(len(y)))
        pickle.dump((x, y, sample_x, sample_y), open('dataset/preprocessed/train_10_05_dataset_1_full.pickle', 'wb'))

        sample_pairs, test_y = generate_test_clustered_pairs(test)
        print('the number of testing samples：' + str(len(test_y)))
        pickle.dump((sample_pairs, test_y), open('dataset/preprocessed/test_10_05_dataset_1_full.pickle', 'wb'))

        replace_samples = generate_test_interpret_dataset(sample_pairs)
        pickle.dump((replace_samples, test_y), open('dataset/preprocessed/test_interpret_10_05_dataset_1_full.pickle', 'wb'))











