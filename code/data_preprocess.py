import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import re
from datetime import datetime

def get_data(data_path):
    """
    :param data_path: 数据路径
    :return: 数据
    """
    data = pd.read_csv(data_path, index_col=0)
    # data.columns = ['id', 'text', 'label']
    return data

def get_continues(data_size, data_pos, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    continual_data = pd.DataFrame(columns=['patch num','year', 'size', 'pos_x', 'pos_y', 'growth_rate'])
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
          
            for k, year in enumerate(years[1:2]):
                index_year = years.get_loc(year)
                growth_rate = temp_row[year] / temp_row[years[index_year - 1]]
                size = temp_row[year]
                year = str(year)         
                pos_x = data_pos.loc[row, str(year) + 'x']
                pos_y = data_pos.loc[row, str(year) + 'y']
                if growth_rate > 0 and growth_rate < 10:
                    continual_data = continual_data.append({'patch num': patch_num ,'year': year, 'size': size, 'pos_x': pos_x, 'pos_y': pos_y,\
                                                         'growth_rate': growth_rate}, ignore_index=True)
            patch_num += 1
    continual_data.sort_values(by=['year', 'growth_rate'], inplace=True)
    info_total = continual_data.groupby('year').agg({'growth_rate': ['mean', 'std', 'count']})
    info_total.to_excel(os.path.join(save_dir, 'info_total.xlsx'))
    continual_data.to_excel(os.path.join(save_dir, 'continual_data.xlsx'))
    return continual_data




if __name__ == '__main__':
    pos = r'231002_reproduce\raw_data\result_pos1229.csv'
    size = r'231002_reproduce\raw_data\result_size1229.csv'
    pos_data = get_data(pos)
    size_data = get_data(size)
    pos_data.replace(-1, np.nan, inplace=True)
    size_data.replace(-1, np.nan, inplace=True)
    save_dir = r'231002_reproduce\preprocess_data'
    continual_data = get_continues(size_data, pos_data, save_dir)

    print(pos_data.head())