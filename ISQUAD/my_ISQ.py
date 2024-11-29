import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import numpy as np


from sklearn.neighbors import KDTree


from ISQUAD.my_anomaly_detection import robust_threshold, t_test

kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]
kpi_type = {1:[_ for _ in range(1,6)], 2:[_ for _ in range(6,8)], 3:[_ for _ in range(8,9)], 4:[_ for _ in range(9,11)],
            5:[_ for _ in range(12,16)], 6:[_ for _ in range(16,18)], 7:[_ for _ in range(18,21)],
            8:[_ for _ in range(21,24)], 9:[_ for _ in range(24,26)], 10:[_ for _ in range(26,29)],
            11:[_ for _ in range(29,31)], 12:[_ for _ in range(31,32)], 13:[_ for _ in range(32,34)],
            14:[_ for _ in range(34,37)], 15:[_ for _ in range(37,41)], 16:[_ for _ in range(41,42)],
            17:[_ for _ in range(42,47)], 18:[_ for _ in range(47,52)], 19:[_ for _ in range(52,54)],
            20:[_ for _ in range(54,58)], 21:[_ for _ in range(58,62)], 22:[_ for _ in range(62,67)],
            23:[_ for _ in range(67,71)], 24:[_ for _ in range(71,84)], 25:[_ for _ in range(84,85)],
            26:[_ for _ in range(85,98)], 27:[_ for _ in range(98,111)]}

class ISQUAD:
    def __init__(self, train_samples:list[np.ndarray], train_y:list, test_samples:[np.ndarray], test_y:list):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.train_y = train_y
        self.test_y = test_y

        ##############################
        self._metrics = kpi_list
        self._metrics_2_type: Dict[int, int] = {}
        self._type_2_metrics: Dict[int, list] = {}

        for t,l in kpi_type.items():
            for k in l:
                if k in kpi_list:
                    self._metrics_2_type[k] = t
                    if t not in self._type_2_metrics:
                        self._type_2_metrics[t] = [k]
                    else :
                        self._type_2_metrics[t].append(k)

        ##############################
        self._metric_patterns: list[list] = []  # each tensor in shape (n_metrics, n_faults, )
        self._idx_2_cluster: Dict[int, int] = {}  #
        self._cluster_2_root_cause_gid: Dict[int, int] = {}
        self._cluster_2_indices: Dict[int, List[int]] = {}  #
        self._kd_tree: Optional[KDTree] = None
        self._test_metric_patterns: list[list] = []

        self._y_preds: list = []  # List of prediction list (list of gid)

    def __call__(self):
        self._kpi_anomaly_detection_()
        self._init_cluster_()
        self._topic_()
        # self._post_init_cluster()
        self._label_clusters()
        self._detection_()
        return self._y_preds


    def _kpi_anomaly_detection_(self):

        for i in range(len(self.train_samples)):
            series = self.train_samples[i]
            pattern = []
            for k in range(1, series.shape[1]):
                rt_ret = robust_threshold(series[:,k].flatten().astype(float))
                t_ret = t_test(series[:,k].flatten().astype(float))
                if rt_ret != 0:
                    pattern.append(rt_ret)
                else:
                    pattern.append(t_ret)
            self._metric_patterns.append(pattern)

        for i in range(len(self.test_samples)):
            series = self.test_samples[i]
            pattern = []
            for k in range(1, series.shape[1]):
                rt_ret = robust_threshold(series[:,k].flatten().astype(float))
                t_ret = t_test(series[:,k].flatten().astype(float))
                if rt_ret != 0:
                    pattern.append(rt_ret)
                else:
                    pattern.append(t_ret)
            self._test_metric_patterns.append(pattern)


    # @profile
    # def _dependency_cleansing_(self):
    #     remove_indices = set()
    #     threshold = 0.95
    #     combinations = list(itertools.combinations(range(len(self._metrics)), 2))
    #     for i, j in tqdm(
    #             combinations, desc='dependency_cleansing'
    #     ):
    #         both = th.count_nonzero(th.logical_and(
    #             self._metric_patterns[i, :self._train_len] != 0, self._metric_patterns[j, :self._train_len] != 0
    #         ))
    #         given_i = th.count_nonzero(self._metric_patterns[i, :self._train_len] != 0)
    #         given_j = th.count_nonzero(self._metric_patterns[j, :self._train_len] != 0)
    #         if both == 0:
    #             continue
    #         elif both / given_i > threshold:
    #             remove_indices.add(j)
    #         elif both / given_j > threshold:
    #             remove_indices.add(i)
    #         else:
    #             continue
    #     indices = sorted(list(set(range(len(self._metrics))) - remove_indices))
    #     self._metrics = [_ for idx, _ in enumerate(self._metrics) if idx not in remove_indices]
    #     self._metric_patterns = self._metric_patterns[indices, :]
    #     logger.debug(f"Cleansed metric pattern size: {self._metric_patterns.size()}")


    def  _init_cluster_(self):
        self._cluster_2_indices = {i: [i] for i in range(len(self.train_samples))}
        self._kd_tree = KDTree(np.array(self._metric_patterns))


    # def _post_init_cluster(self):
    #     for cluster, indices in self._cluster_2_indices.items():
    #         for idx in indices:
    #             self._idx_2_cluster[idx] = cluster


    def _topic_(self):

        changed = True
        while changed:
            changed = False
            for cluster, fault_indices in list(self._cluster_2_indices.items()):
                # if len(fault_indices) > 1:
                #     continue
                i = fault_indices[0]
                # print(self._kd_tree.query(np.array(self._metric_patterns[i]).reshape(1, -1), 2))
                j = self._kd_tree.query(np.array(self._metric_patterns[i]).reshape(1, -1), 2)[1][0]

                if j[0] != i:
                    j = j[0]
                else:
                    j = j[1]
                assert i != j
                if i not in self._cluster_2_indices or j not in self._cluster_2_indices:
                    continue
                # print(self.similarity(self._metric_patterns[i], self._metric_patterns[j]))
                if self.similarity(self._metric_patterns[i], self._metric_patterns[j]) > 1:
                    changed = True
                    if len(self._cluster_2_indices[i]) > len(self._cluster_2_indices[j]):
                        self._cluster_2_indices[i] += self._cluster_2_indices[j]
                        del self._cluster_2_indices[j]
                    else:
                        self._cluster_2_indices[j] += self._cluster_2_indices[i]
                        del self._cluster_2_indices[i]
        for cluster, indices in self._cluster_2_indices.items():
            for idx in indices:
                self._idx_2_cluster[idx] = cluster

    def similarity(self, pattern1: list, pattern2: list):
        type_sum = defaultdict(float)
        for i, metric in enumerate(self._metrics):
            type_sum[self._metrics_2_type[metric]] += pattern1[i] == pattern2[i]
        for k in type_sum.keys():
            type_sum[k] = type_sum[k] / len(self._type_2_metrics[k])

        v = list(type_sum.values())
        return np.sqrt(np.sum(np.square(v)) / len(v))

    def _detection_(self):

        assert self._kd_tree is not None
        for i in range(len(self.test_samples)):
            pattern = self._test_metric_patterns[i]
            similarities = []
            for j in range(len(self.train_samples)):
                similarities.append(self.similarity(pattern, self._metric_patterns[j]))
            nearest = np.argsort(similarities)[::-1][0]
            self._y_preds.append(self._cluster_2_root_cause_gid[self._idx_2_cluster[nearest]])


    def _label_clusters(self):
        for cluster_idx, fault_indices in self._cluster_2_indices.items():
            train_y = [self.train_y[i] for i in fault_indices]
            rc_counter = Counter(train_y)
            self._cluster_2_root_cause_gid[cluster_idx] = rc_counter.most_common(1)[0][0]

if __name__ == '__main__':
    print(np.argsort([1,3,2,0]))