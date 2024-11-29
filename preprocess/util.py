import datetime
import math
import pickle

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.stats import norm


from scipy import stats


def z_score_normalization(time_series):
    """
    对时间序列进行Z-Score归一化，处理标准差为0的情况

    参数:
    time_series: numpy array, 时间序列数据

    返回:
    归一化的时间序列
    """
    mean = np.mean(time_series)
    std = np.std(time_series)
    if std == 0:
        # 标准差为0时返回一个全零序列
        # 或者，根据需求可能返回原序列: return time_series
        return np.zeros_like(time_series)
    else:
        normalized_time_series = (time_series - mean) / std
        return normalized_time_series


def min_max_normalization(time_series, new_min=0, new_max=1):
    """
    对时间序列进行Min-Max归一化

    参数:
    time_series: numpy array, 时间序列数据
    new_min: 归一化后的最小值，默认为0
    new_max: 归一化后的最大值，默认为1

    返回:
    归一化的时间序列
    """
    # 计算原始时间序列的最小值和最大值
    original_min = np.min(time_series)
    original_max = np.max(time_series)

    # 避免除以0的情况，如果原始序列的最小值和最大值相同，则返回原序列或其他处理
    if original_min == original_max:
        # 可以选择返回一个全为new_min或new_max的序列，这里选择返回new_min
        return np.full_like(time_series, new_min)
    else:
        # 进行Min-Max归一化
        normalized_time_series = (time_series - original_min) / (original_max - original_min) * (
                new_max - new_min) + new_min
        return normalized_time_series



def normalize(data, mode=1):
    new_data = []
    for ts in data:
        new_ts = np.copy(ts)
        for j in range(1, new_ts.shape[1]):
            if mode == 1:
                new_ts[:, j] = z_score_normalization(new_ts[:, j])
            elif mode == 2:
                new_ts[:, j] = min_max_normalization(new_ts[:, j])
        new_data.append(new_ts)
    return new_data


def softmax(x):
    # # 计算每行的最大值
    row_max = np.max(x)
    # # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)

    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s


def sq_weight(x):
    x_sq = np.square(x)
    x_sum = np.sum(x_sq)
    s = x_sq / x_sum
    return s


def calculate_entropy(distribution):
    """
    计算给定概率分布的信息熵。

    :param distribution: 分布列表
    :return: 该概率分布的信息熵
    """

    probabilities = distribution / np.sum(distribution)
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)  # 使用以2为底的对数计算信息熵，单位是比特(bit)
    return entropy


def back_softmax(x):
    # # 计算每行的最大值
    # row_max = np.max(x)
    # # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    # x = x - row_max
    # 计算e的指数次幂

    x_exp = np.exp(np.array(x) * (-1))

    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s


def inverse_weight(x):
    x_inverse = np.reciprocal(x)
    x_sum = np.sum(x_inverse)
    s = x_inverse / x_sum
    return s


def cauchy_cdf(x, median, mad):
    t = np.arctan((x - median)/mad)
    return t/np.pi + 0.5


def cauchy_tansform1(x, median):
    if x >= median:
        return x
    t = np.tan(np.pi*(x - median)/(2*median))
    return 2*median*t/np.pi + median


def cauchy_tansform2(x, median, mad):
    if x >= 0:
        return x
    t = x * mad / median * np.pi / 2
    return np.tan(t)


def kpi_cdf1(x, median, mad):
    return cauchy_cdf(cauchy_tansform1(x, median), median, mad)


def kpi_cdf2(x, median, mad):
    return cauchy_cdf(cauchy_tansform2(x, median, mad), 0, 1)


def kpi_cdf3(x):
    return cauchy_cdf(x, 0, 1)


def calculate_median_mad(ts):
    ts = np.array(ts)
    median = np.median(ts)
    ts = np.absolute(ts - median)
    mad = np.median(ts)
    return median, mad


def wasserstein_dist(p, q):
    pos = [-1, 0, 1]
    return scipy.stats.wasserstein_distance(pos, pos, p, q)


def check_shift(data):
    break_point = len(data) // 2
    statistic, p_value = stats.ttest_ind(data[break_point:], data[:break_point])
    print(statistic, p_value)
    return statistic, p_value

def sbd_to_norm(sbd_dist):
    t = 1 - sbd_dist/2
    return -math.log(t)

def sbd_norm_pdf(norm, sbd_dist):
    pdf = norm.pdf(sbd_to_norm(sbd_dist))
    return 2 * pdf


# 定义通道独立的欧氏距离度量
def independent_channel_euclidean(X, Y):
    """计算两个多元时间序列之间的通道独立欧氏距离"""
    assert X.shape == Y.shape, "时间序列必须具有相同的形状"

    n_channels = X.shape[1]
    total_distance = 0.0

    # 对每个通道独立计算距离
    for c in range(n_channels):
        # 计算当前通道上的欧氏距离
        channel_distance = np.sqrt(np.sum((X[:, c] - Y[:, c]) ** 2))
        total_distance += channel_distance

    return total_distance