import math

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def eucli_dist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def compute_medoid(vectors):
    """
    计算多个向量的 medoid.

    参数:
    vectors: (N, M) 的数组，表示 N 个 M 维向量

    返回:
    medoid: 距离其他向量距离和最小的向量
    """
    n_series = len(vectors)
    distance_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(i + 1, n_series):
            distance = eucli_dist(vectors[i], vectors[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance


    # 计算每个向量与其他向量的距离和
    distance_sums = np.sum(distance_matrix, axis=1)

    # 距离和最小的索引
    medoid_index = np.argmin(distance_sums)

    # 返回 medoid 向量
    return vectors[medoid_index], medoid_index

def print2dlist(list):
    for i in range(len(list)):
        print(list[i])



def printresult(groundtruth, pred):
    acc = accuracy_score(groundtruth, pred)
    prf = precision_recall_fscore_support(groundtruth, pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    average_metric = [0, 0, 0]

    print('Acc = {:.4f}'.format(acc))

    # P, R and F1 of each fault
    for j in range(9):
        print('{:<25}\t{:.4f},{:.4f},{:.4f}'.format(
            j, prf[0][j], prf[1][j], prf[2][j]))
        average_metric[0] += prf[0][j]
        average_metric[1] += prf[1][j]
        average_metric[2] += prf[2][j]

    print("average P:" + str(average_metric[0] / 9))
    print("average R:" + str(average_metric[1] / 9))
    print("average F1:" + str(average_metric[2] / 9))


def get_prediction(test_sample_embeddings, train_sample_embeddings, train_labels):
    pred = []
    for i in range(len(test_sample_embeddings)):
        # for i in range(1):
        test_sample_embedding = test_sample_embeddings[i]
        dist = []
        for j in range(len(test_sample_embedding)):
            t = []
            for m in range(len(train_sample_embeddings)):
                train_sample_embedding = train_sample_embeddings[m]
                for n in range(len(train_sample_embedding)):
                    t.append(eucli_dist(test_sample_embedding[j], train_sample_embedding[n]))
            dist.append(t)
        # print2dlist(dist)
        # print([e.item() for e in train_labels])
        # print(test_sample_labels[i].item())
        dist = np.array(dist)
        # 获取展平数组中的最小值索引
        min_index_flat = np.argmin(dist)
        # 将展平索引转换为多维数组的坐标
        min_index_2d = np.unravel_index(min_index_flat, dist.shape)
        # print("最小值的二维索引:", min_index_2d)
        pred.append(train_labels[min_index_2d[1]])
    return pred


def aggregate_train_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                   agg_method='median'):
    pred = []
    for i in range(len(test_sample_embeddings)):
        # for i in range(1):
        test_sample_embedding = test_sample_embeddings[i]
        dist = []
        for j in range(len(test_sample_embedding)):
            t = []
            for m in range(len(train_sample_embeddings)):
                if agg_method == 'median':
                    # aggregate_embedding = np.median(np.array(train_sample_embeddings[m]), axis=0).tolist()
                    aggregate_embedding = compute_medoid(train_sample_embeddings[m])[0]
                else:
                    aggregate_embedding = np.average(np.array(train_sample_embeddings[m]), axis=0).tolist()
                t.append(eucli_dist(test_sample_embedding[j], aggregate_embedding))
            dist.append(t)
        # print2dlist(dist)
        # print([e.item() for e in train_labels])
        # print(test_sample_labels[i].item())
        dist = np.array(dist)
        # 获取展平数组中的最小值索引
        min_index_flat = np.argmin(dist)
        # 将展平索引转换为多维数组的坐标
        min_index_2d = np.unravel_index(min_index_flat, dist.shape)
        # print("最小值的二维索引:", min_index_2d)
        pred.append(train_sample_labels[min_index_2d[1]])
    return pred


def aggregate_test_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                  agg_method='median'):
    pred = []
    for i in range(len(test_sample_embeddings)):
        # for i in range(1):
        test_sample_embedding = test_sample_embeddings[i]
        if agg_method == 'median':
            # aggregate_embedding = np.median(np.array(test_sample_embedding), axis=0).tolist()
            aggregate_embedding = compute_medoid(test_sample_embedding)[0]
        else:
            aggregate_embedding = np.average(np.array(test_sample_embedding), axis=0).tolist()
        dist = []
        for j in range(len(train_sample_embeddings)):
            t = []
            train_sample_embedding = train_sample_embeddings[j]
            for m in range(len(train_sample_embedding)):
                t.append(eucli_dist(aggregate_embedding, train_sample_embedding[m]))
            dist.append(t)
        # print2dlist(dist)
        # print([e.item() for e in train_labels])
        # print(test_sample_labels[i].item())
        dist = np.array(dist)
        # 获取展平数组中的最小值索引
        min_index_flat = np.argmin(dist)
        # 将展平索引转换为多维数组的坐标
        min_index_2d = np.unravel_index(min_index_flat, dist.shape)
        # print("最小值的二维索引:", min_index_2d)
        pred.append(train_sample_labels[min_index_2d[0]])
    return pred


def aggregate_train_test_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels,
                                        agg_method='median'):
    pred = []
    for i in range(len(test_sample_embeddings)):
        # for i in range(1):
        test_sample_embedding = test_sample_embeddings[i]
        if agg_method == 'median':
            # aggregate_test_embedding = np.median(np.array(test_sample_embedding), axis=0).tolist()
            aggregate_test_embedding = compute_medoid(test_sample_embedding)[0]
        else:
            aggregate_test_embedding = np.average(np.array(test_sample_embedding), axis=0).tolist()
        dist = []
        for j in range(len(train_sample_embeddings)):
            train_sample_embedding = train_sample_embeddings[j]
            if agg_method == 'median':
                # aggregate_train_embedding = np.median(np.array(train_sample_embedding), axis=0).tolist()
                aggregate_train_embedding = compute_medoid(train_sample_embedding)[0]
            else:
                aggregate_train_embedding = np.average(np.array(train_sample_embedding), axis=0).tolist()

            dist.append(eucli_dist(aggregate_test_embedding, aggregate_train_embedding))
        # print2dlist(dist)
        # print([e.item() for e in train_labels])
        # print(test_sample_labels[i].item())
        dist = np.array(dist)
        # 获取展平数组中的最小值索引
        min_index = np.argmin(dist)

        # print("最小值的二维索引:", min_index_2d)
        pred.append(train_sample_labels[min_index])
    return pred

def election_based_get_prediction(test_sample_embeddings, train_sample_embeddings, train_sample_labels, agg_method='median'):
    pred = []
    for i in range(len(test_sample_embeddings)):
        # for i in range(1):
        test_sample_embedding = test_sample_embeddings[i]
        k_pred = []
        for j in range(len(test_sample_embedding)):
            dist = []
            for m in range(len(train_sample_embeddings)):
                if agg_method == 'median':
                    # aggregate_embedding = np.median(np.array(train_sample_embeddings[m]), axis=0).tolist()
                    aggregate_embedding = compute_medoid(train_sample_embeddings[m])[0]
                else:
                    aggregate_embedding = np.average(np.array(train_sample_embeddings[m]), axis=0).tolist()
                dist.append(eucli_dist(test_sample_embedding[j], aggregate_embedding))
            dist = np.array(dist)
            min_index = np.argmin(dist)
            k_pred.append(train_sample_labels[min_index])
        # print(stats.mode(k_pred, keepdims=False))
        mode = stats.mode(k_pred, keepdims=False)[0]
        count = stats.mode(k_pred, keepdims=False)[1]
        pred.append(mode)
        # if count >= len(k_pred) / 2:
        #     pred.append(mode)
        # else:
        #     pred.append(9)
    return pred
