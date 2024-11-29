import copy

import numpy as np

from scipy.stats import  stats


def robust_threshold(
        series: np.ndarray, upper_limit: float = 0.97, lower_limit: float = 0.03, m: float = 1,
) -> int:
    pi = 3.1415926

    median = np.median(series)
    mad = np.median(np.abs(series - median))  # (...)
    t_series = copy.deepcopy(series)

    t_series[series < m] = 2 * m / pi * np.tan(pi * (series[series < m] - m) / (2 * m)) + m

    cdf = np.arctan((t_series - median) / np.clip(mad, a_min=1e-4, a_max=None)) / pi + 0.5

    ret = np.zeros_like(series)
    ret[cdf > upper_limit] = 1
    ret[cdf < lower_limit] = -1

    s = np.sum(ret)
    if s >= len(series)//3:
        return 1
    elif s <= -len(series)//3:
        return -1
    else:
        return 0


def t_test(
        series: np.ndarray, significance_level: float = 0.05,
) -> int:
    break_point = len(series) // 2
    statistic, p_value = stats.ttest_ind(series[break_point:], series[:break_point])
    # return statistic, p_value
    if p_value < significance_level:
        if statistic > 0:
            return 2
        else:
            return -2
    else:
        return 0

