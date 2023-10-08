import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import math
from scipy.spatial import distance

def get_min_distance(data, X, Y, year, r_area, index, n=5):
    # use edge detector to extract nearest edges (include independent patches)
    x = X[index]
    y = Y[index]
    year = year[index]
    r_list = r_area[index]
    need_year = np.arange(2015, year + 1)
    need_data = data[data['year'].isin(need_year)]
    x_list = need_data['pos_x'].values
    y_list = need_data['pos_y'].values
    r_area = need_data['r_area'].values
    # 使用广播计算两个点和两个数组之间的距离
    point = np.array((x, y))
    points_list = np.column_stack((x_list, y_list))
    dist = distance.cdist([point], points_list) - r_area - r_list

    # 找到前n个最小的距离
    min_distances = np.partition(dist, range(n), axis=None)[:n]

    # 第n近的距离是 min_distances[n-1]
    nth_min_distance = min_distances[1:n - 1].mean()

    return nth_min_distance



if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather.xlsx'
    data = pd.read_excel(data_path)
    X = data['pos_x'].values
    Y = data['pos_y'].values
    year = data['year'].values
    size = data['size'].values
    r_area = size ** 0.5 / math.pi
    data['r_area'] = r_area
    for i in tqdm(range(len(X))):
        min_distance = get_min_distance(data, X, Y, year, r_area, i)
        data.loc[i, 'min_distance'] = min_distance
    data.to_excel(r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather_dist.xlsx', index=False)
