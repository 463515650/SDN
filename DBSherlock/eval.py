import pickle

import numpy as np

from DBSherlock.anomaly_data import AnomalyDataset, AnomalyData
from DBSherlock.causal_model import CausalModel
from DBSherlock.dbsherlock import DBSherlock
from util import printresult

kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]

if __name__ == '__main__':


    train_x, train_y, train_cluster_index, test_x, test_y, test_cluster_index, normal_data  = pickle.load(open(r'../dbsherlock_train_test_10_05_dataset.pickle', 'rb'))

    train = False

    concated_normal = []

    # Concatenate the normal data together
    for cluster in normal_data:
        t = np.concatenate(cluster, axis=0)
        t = t[:,kpi_list].tolist()
        concated_normal.append(t)


    # The test abnormal sample data set was constructed
    anomaly_data = []
    for i in range(len(test_x)):
        x = test_x[i]
        x = x[:, kpi_list].tolist()
        y = test_y[i]
        normal_index = test_cluster_index[i]
        data_dict = {'cause': y, 'attributes': kpi_list, 'values': x + concated_normal[normal_index],
                     'normal_regions': [_ for _ in range(len(x), len(x) + len(concated_normal[normal_index]))],
                     'abnormal_regions': [_ for _ in range(len(x))]}
        anomaly_data.append(AnomalyData.from_dict(data=data_dict))

    test_anomaly_dataset = AnomalyDataset.from_dict(data={'causes': list(set(test_y)), 'data': anomaly_data})

    dbsherlock = DBSherlock()

    if train:
        # The training abnormal sample data set is constructed
        anomaly_data = []
        for i in range(len(train_x)):
            x = train_x[i]
            x = x[:,kpi_list].tolist()
            y = train_y[i]
            normal_index = train_cluster_index[i]
            data_dict = {'cause':y, 'attributes':kpi_list, 'values':x + concated_normal[normal_index],
                         'normal_regions': [_ for _ in range(len(x), len(x) + len(concated_normal[normal_index]))],
                         'abnormal_regions': [_ for _ in range(len(x))]}
            anomaly_data.append(AnomalyData.from_dict(data=data_dict))

        train_anomaly_dataset = AnomalyDataset.from_dict(data={'causes':list(set(train_y)), 'data':anomaly_data})


        merged_causal_model_per_cause = {}

        for e in train_anomaly_dataset.causes:
            train_data_per_cause = train_anomaly_dataset.get_data_of_cause(e)

            causal_models = []
            for training_data in train_data_per_cause:
                causal_models.append(dbsherlock.create_causal_model(data=training_data))
                # print(causal_models[-1])
                print('ok')
            merged_causal_model_per_cause[e] = sum(causal_models)
            merged_causal_model_per_cause[e].save(path=f'{e}_causal_model.json')
    else:
        merged_causal_model_per_cause = {}
        for e in test_anomaly_dataset.causes:
            merged_causal_model_per_cause[e] = CausalModel(None, None).load(path=f'{e}_causal_model.json')

    # Diagnose testing set samples
    pred = []
    for i in range(len(test_anomaly_dataset)):
        testing_data = test_anomaly_dataset[i]
        confidence = {}
        for cause, model in merged_causal_model_per_cause.items():
            confidence[cause] = dbsherlock.compute_confidence(causal_model=model, data=testing_data)[0]
        print(confidence)
        # 使用 max 和 lambda 找出 value 最大的 key
        max_cause = max(confidence, key=lambda k: confidence[k])
        pred.append(max_cause)
    printresult(test_y, pred)








