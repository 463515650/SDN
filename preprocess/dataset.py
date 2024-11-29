import pickle
import random

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

import preprocess.util
from preprocess import util


class dataset:
    def __init__(self, path):
        self.path = path
        # Stores normal data
        self.normal_data = []
        # Stores labels for anomaly occurrence periods
        self.labels = []
        # Raw KPI data
        self.raw_data = []
        # Anomalous data after window segmentation
        self.aligned_data = []
        # Normal data after window segmentation
        self.aligned_normal = []
        # # Data after Z-score normalization of windowed data
        # self.z_score_data = []
        # # Data after Min-Max normalization of windowed data
        # self.min_max_data = []
        # Cluster centroids after clustering normal data
        self.clustered_normal_index = []
        # Anomaly type labels
        self.anomaly_type = ['fault1', 'fault2', 'fault3', 'fault4', 'fault5', 'lockwait', 'multiindex', 'setknob', 'stress']


    # Load data of a single anomaly type from file
    def load_file(self, path):
        f = open(path, 'rb')
        data = pickle.load(f)
        ts = []
        label = []
        for i in range(0, len(data)):
            ts.append(np.array(data[i][0]))
            label.append(data[i][1])
        return ts, label

    # Load the entire dataset from file
    def load_data(self):
        self.normal_data = self.load_file(self.path + r'\normal_data.pickle')[0]
        for t in self.anomaly_type:
            anomaly_data, label = self.load_file(self.path + '\\' + t + r'_data.pickle')
            self.raw_data.append(anomaly_data)
            self.labels.append(label)
            # self.z_score_data.append(util.normalize(anomaly_data, 1))
            # self.min_max_data.append(util.normalize(anomaly_data, 2))


    # Align the original data by segmenting it into time windows
    def align_data(self):
        for i in range(len(self.anomaly_type)):
            data = []
            if self.anomaly_type[i] == 'setknob':
                for ts in self.raw_data[i]:
                    s = 0
                    while s + 10 < len(ts):
                        data.append(ts[s + 1:s + 1 + 10])
                        s += 12
            elif self.anomaly_type[i] == 'multiindex':
                for ts in self.raw_data[i]:
                    data.append(ts[1:1 + 10])
            else:
                for j in range(len(self.raw_data[i])):
                    data.append(self.raw_data[i][j][self.labels[i][j][0] - 1:self.labels[i][j][0] - 1 + 10])
            self.aligned_data.append(data)
            # self.z_score_data.append(util.normalize(data, 1))
            # self.min_max_data.append(util.normalize(data, 2))
        for i in range(len(self.normal_data)):
            self.aligned_normal.append(self.normal_data[i][1:1+10])


    # Split into training and testing sets, with test set ratio defined by 'ratio'
    def split(self, ratio):
        train = dataset(self.path)
        test = dataset(self.path)
        for i in range(len(self.anomaly_type)):
            perm = [r for r in range(len(self.aligned_data[i]))]
            random.shuffle(perm)
            test_id = set(perm[:int(len(self.aligned_data[i]) * ratio)])
            ts_train = []
            ts_test = []
            label_train = []
            label_test = []
            for j in range(len(self.aligned_data[i])):
                if j in test_id:
                    ts_test.append(self.aligned_data[i][j])
                    label_test.append(self.labels[i][j])
                else:
                    ts_train.append(self.aligned_data[i][j])
                    label_train.append(self.labels[i][j])
            test.aligned_data.append(ts_test)
            test.labels.append(label_test)
            train.aligned_data.append(ts_train)
            train.labels.append(label_train)
        test.normal_data = self.normal_data
        train.normal_data = self.normal_data
        test.aligned_normal = self.aligned_normal
        train.aligned_normal = self.aligned_normal
        train.clustered_normal_index = self.clustered_normal_index
        test.clustered_normal_index = self.clustered_normal_index
        # test.align_data()
        # train.align_data()
        # test.model_normal(self.kpi_list)
        # train.model_normal(self.kpi_list)

        return train, test

    def clip(self, anomaly_nums, normal_nums):
        selected = np.random.choice(len(self.normal_data), size=normal_nums, replace=False)
        t = []
        for i in selected:
            t.append(self.normal_data[i])
        self.normal_data = t
        t = []
        for i in selected:
            t.append(self.aligned_normal[i])
        self.aligned_normal = t
        for i in range(len(self.anomaly_type)):
            selected = np.random.choice(len(self.aligned_data[i]), size=anomaly_nums, replace=False)
            t = []
            for j in selected:
                t.append(self.aligned_data[i][j])
            self.aligned_data[i] = t


    def cluster_normal(self, k, kpi_list):
        n_series = len(self.aligned_normal)
        distance_matrix = np.zeros((n_series, n_series))

        for i in range(n_series):
            for j in range(i + 1, n_series):
                distance = util.independent_channel_euclidean(self.aligned_normal[i][:,kpi_list], self.aligned_normal[j][:,kpi_list])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Use K-Medoids clustering
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
        labels = kmedoids.fit_predict(distance_matrix)
        # Output the results
        print("Clustering labels:", labels)
        # Output the cluster centroids
        self.clustered_normal_index = kmedoids.medoid_indices_  # Get the indices of the centroids

class multidataset:
    def __init__(self, datasets):
        self.datasets = datasets
        # Store the index of data in the dataset
        self.index = []
        # Store the index of normal data in the dataset
        self.normal_index = []
        # Cluster centroids of normal data
        self.clustered_normal_index = []
        # Store labels for anomaly occurrence periods
        self.labels = []
        # Aligned normal data
        self.aligned_normal = []
        # Raw KPI data
        # self.raw_data = []
        # KPI data after window segmentation
        self.aligned_data = []
        # Data after Z-score normalization of windowed data
        # self.z_score_data = []
        # All anomaly types' full data
        # self.full_z_score_data = []
        # Data after Min-Max normalization of windowed data
        # self.min_max_data = []
        # # Discrete feature of the sample's shape
        # self.pattern_feature = []
        # # Value anomaly feature of the sample
        # self.value_feature = []
        # Anomaly type labels
        self.anomaly_type = ['fault1', 'fault2', 'fault3', 'fault4', 'fault5', 'lockwait', 'multiindex', 'setknob',
                             'stress']
        self.normalized_feature = []
        self.normalized_normal = []


    def load(self):
        for i in range(len(self.anomaly_type)):
            index = []
            labels = []
            # raw_data = []
            aligned_data = []
            # z_score_data = []
            # min_max_data = []
            for j in range(len(self.datasets)):
                index.extend([j] * len(self.datasets[j].aligned_data[i]))
                labels.extend(self.datasets[j].labels[i])
                # raw_data.extend(self.datasets[j].raw_data[i])
                aligned_data.extend(self.datasets[j].aligned_data[i])
                # z_score_data.extend(self.datasets[j].z_score_data[i])
                # min_max_data.extend(self.datasets[j].min_max_data[i])
                # self.full_z_score_data.extend(self.datasets[j].z_score_data[i])
            self.index.append(index)
            self.labels.append(labels)
            # self.raw_data.append(raw_data)
            self.aligned_data.append(aligned_data)
            # self.z_score_data.append(z_score_data)
            # self.min_max_data.append(min_max_data)
        for i in range(len(self.datasets)):
            self.aligned_normal.extend(self.datasets[i].aligned_normal)
            self.normal_index.extend([i] * len(self.datasets[i].aligned_normal))
        offset = 0
        for i in range(len(self.datasets)):
            self.clustered_normal_index.append([e+offset for e in self.datasets[i].clustered_normal_index])
            offset += len(self.datasets[i].aligned_normal)

    def normalize(self):
        self.normalized_feature = []
        allfea = []
        for i in range(len(self.anomaly_type)):
            for j in range(len(self.aligned_data[i])):
                allfea.append(self.aligned_data[i][j][:, 1:])
                # if self.aligned_data[i][j].shape[0] != 10:
                #     print(str(i) + ',' + str(j) + ' error')
        for i in range(len(self.aligned_normal)):
            allfea.append(self.aligned_normal[i][:, 1:])
        allfea = np.array(allfea)
        metric_cnt = allfea.shape[2]
        allfea = allfea.reshape([-1, metric_cnt])
        transform = StandardScaler()
        normed = transform.fit_transform(allfea)
        result = list(normed.reshape([-1, 10, metric_cnt]))
        offset = 0
        for i in range(len(self.anomaly_type)):
            self.normalized_feature.append(result[offset:offset + len(self.aligned_data[i])])
            offset += len(self.aligned_data[i])
        self.normalized_normal = result[offset:]

    @staticmethod
    def joint_normalize(train, test):
        train.normalized_feature = []
        test.normalized_feature = []
        allfea = []
        for i in range(len(train.anomaly_type)):
            for j in range(len(train.aligned_data[i])):
                allfea.append(train.aligned_data[i][j][:, 1:])

        for i in range(len(test.anomaly_type)):
            for j in range(len(test.aligned_data[i])):
                allfea.append(test.aligned_data[i][j][:, 1:])
                # if self.aligned_data[i][j].shape[0] != 10:
                #     print(str(i) + ',' + str(j) + ' error')
        for i in range(len(train.aligned_normal)):
            allfea.append(train.aligned_normal[i][:, 1:])
        allfea = np.array(allfea)
        metric_cnt = allfea.shape[2]
        allfea = allfea.reshape([-1, metric_cnt])
        transform = StandardScaler()
        normed = transform.fit_transform(allfea)
        print(allfea.shape)
        result = list(normed.reshape([-1, 10, metric_cnt]))
        offset = 0
        for i in range(len(train.anomaly_type)):
            train.normalized_feature.append(result[offset:offset + len(train.aligned_data[i])])
            offset += len(train.aligned_data[i])
        for i in range(len(train.anomaly_type)):
            test.normalized_feature.append(result[offset:offset + len(test.aligned_data[i])])
            offset += len(test.aligned_data[i])
        train.normalized_normal = result[offset:]
        test.normalized_normal = result[offset:]

    # def smooth(self, window_size):
    #     for

if __name__ == '__main__':
    path = [r'C:\Users\yls\Desktop\diagnosis\DBPA_dataset-main\32_128\single',
            r'C:\Users\yls\Desktop\diagnosis\DBPA_dataset-main\32_256\single',
            r'C:\Users\yls\Desktop\diagnosis\DBPA_dataset-main\64_128\single',
            r'C:\Users\yls\Desktop\diagnosis\DBPA_dataset-main\64_256\single']
    out = [r'dataset\32_128.data',
           r'dataset\32_256.data',
           r'dataset\64_128.data',
           r'dataset\64_256.data']

    for i in range(len(path)):
        data = dataset(path[i])
        data.load_data()
        data.align_data()
        pickle.dump(data, open(out[i], 'wb'))