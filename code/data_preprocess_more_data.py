import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import re
from datetime import datetime


src_points = np.array([[0, 0], [15000, 18000]])
dst_points = np.array([[117.414905, 23.933933], [117.430231, 23.915684]])

def src2dst(x,y):
    x = (x - src_points[0][0]) / (src_points[1][0] - src_points[0][0]) * (dst_points[1][0] - dst_points[0][0]) + dst_points[0][0]
    y = (y - src_points[0][1]) / (src_points[1][1] - src_points[0][1]) * (dst_points[1][1] - dst_points[0][1]) + dst_points[0][1]
    return x,y

def dst2src(x,y):
    x = (x - dst_points[0][0]) / (dst_points[1][0] - dst_points[0][0]) * (src_points[1][0] - src_points[0][0]) + src_points[0][0]
    y = (y - dst_points[0][1]) / (dst_points[1][1] - dst_points[0][1]) * (src_points[1][1] - src_points[0][1]) + src_points[0][1]
    return x,y

def get_data(data_path):
    """
    :param data_path: 数据路径
    :return: 数据
    """
    data = pd.read_csv(data_path, index_col=0)
    # data.columns = ['id', 'text', 'label']
    return data

def get_date(string, pattern):
    match = re.search(pattern, string)
    if match:
        date_part = match.group()
        if len(date_part) == 6:
            date_format = '%Y%m'
        elif len(date_part) == 8:
            date_format = '%Y%m%d'
        elif len(date_part) == 4:
            date_format = '%Y'
        else:
            raise ValueError(f"无法解析日期部分：{date_part}")
        date_obj = datetime.strptime(date_part, date_format)
        return date_obj
    else:
        return None
    
def get_weather_data(weather_path, data, start_feature = 'Average temperature (℃)', end_feature = 'Average temperature from December to February (℃)', save_dir = None):
    if os.path.basename(weather_path).split('.')[1] == 'xlsx' or os.path.basename(weather_path).split('.')[1] == 'xls':
        weather_data = pd.read_excel(weather_path)
    elif os.path.basename(weather_path).split('.')[1] == 'csv':
        weather_data = pd.read_csv(weather_path)
    else:
        raise ValueError('weather data format is not supported')
    weather_data.dropna(axis=0, how='all', inplace=True)
    year = weather_data['Year']
    weather_data = weather_data.loc[:, start_feature:end_feature]
    year2feature = {}
    for i in weather_data.columns:
        feature = dict(zip(year, weather_data[i]))
        year2feature[i] = feature

    for feature_name, feature_data in year2feature.items():
        year = data['year']
        data[feature_name] = year.map(feature_data)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data.to_excel(os.path.join(save_dir, 'data_with_weather.xlsx'))
    return data
            


    

def get_continues(data_size, data_pos, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    continual_data = pd.DataFrame(columns=['patch num','file name','year', 'size', 'pos_x', 'pos_y', 'growth_rate', 'growth days'])
    try:
        data_size.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    w, h = data_size.shape
    patch_num = 0
    pattern1 = r'(\d{8}|\d{6})' # 定义一个正则表达式模式来匹配日期部分
    pattern2 = r'(\d{4})'
    for row in tqdm(range(w)):
        temp_row = data_size.iloc[row, :]
        temp_row = temp_row.dropna()
        if len(temp_row) <= 1:
            continue
        else:
            years = temp_row.index
            year_list = []
            date_list = []
            for s in years:
                year_list.append(get_date(s, pattern2))
                date_list.append(get_date(s, pattern1))

            for k, year in enumerate(years[1:]):
                index_year = years.get_loc(year)
                growth_days = (date_list[index_year] - date_list[index_year-1]).days
                growth_rate = (temp_row[index_year] - temp_row[index_year -1]) / growth_days
                size = temp_row[year]
                year = str(year)         
                pos_x = data_pos.loc[row, str(year) + 'x']
                pos_y = data_pos.loc[row, str(year) + 'y']
                if growth_rate > 0 and growth_rate < 200:
                    continual_data = continual_data.append({'patch num': patch_num ,'file name': year, 'year': get_date(year, pattern1).year, 'size': size, 'pos_x': pos_x, 'pos_y': pos_y,\
                                                         'growth_rate': growth_rate, 'growth days': growth_days}, ignore_index=True)
            patch_num += 1
    continual_data.sort_values(by=['year', 'growth_rate'], inplace=True)
    continual_data.reset_index(drop=True, inplace=True)
    X, Y = continual_data['pos_y'], 18000  - continual_data['pos_x']
    X, Y = src2dst(X,Y)
    continual_data['lat'], continual_data['lon'] = X, Y
    info_total = continual_data.groupby('year').agg({'growth_rate': ['mean', 'std', 'count']})
    info_total.to_excel(os.path.join(save_dir, 'info_total.xlsx'))
    continual_data.to_excel(os.path.join(save_dir, 'continual_data.xlsx'))
    return continual_data


if __name__ == '__main__':
    pos = r'231002_reproduce\raw_data\result_pos.csv'
    size = r'231002_reproduce\raw_data\result_size.csv'
    pos_data = get_data(pos)
    size_data = get_data(size)
    pos_data.replace(-1, np.nan, inplace=True)
    size_data.replace(-1, np.nan, inplace=True)
    save_dir = r'231002_reproduce\preprocess_data'
    continual_data = get_continues(size_data, pos_data, save_dir)
    weather_path = r'E:\XMU\231002_reproduce\raw_data\xiamen.xlsx'
    data = get_weather_data(weather_path, continual_data, save_dir=save_dir)
    print(pos_data.head())
    