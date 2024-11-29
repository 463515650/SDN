import pickle
import numpy as np

from ISQUAD.my_ISQ import ISQUAD
from util import printresult

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = pickle.load(open(r'../isquad_train_test_10_05_dataset.pickle', 'rb'))
    ISQ = ISQUAD(train_x, train_y, test_x, test_y)
    ISQ()
    # for i in ISQ._metric_patterns:
    #     print(i)

    # print(ISQ._cluster_2_root_cause_gid)
    # print(ISQ._cluster_2_indices)
    # print(ISQ._y_preds)
    # print(test_y)
    printresult(test_y, ISQ._y_preds)


