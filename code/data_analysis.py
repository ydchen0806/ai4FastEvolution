import re
from datetime import datetime
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
import csv

if __name__ == '__main__':
    size_path = r'E:\XMU\231002_reproduce\raw_data\result_size.csv'
    size_total_path = r'E:\XMU\231002_reproduce\raw_data\result_size1229.csv'
    size = pd.read_csv(size_path)
    size_total = pd.read_csv(size_total_path)
    size.replace(-1, np.nan, inplace=True)
    size_total.replace(-1, np.nan, inplace=True)
    date_strings  = size.columns[1:].tolist()
    # 定义一个正则表达式模式来匹配日期部分
    date_pattern = r'(\d{8}|\d{6})'

    previous_date = None
    time_differences = []  # 保存时间差的列表

    for s in date_strings:
        match = re.search(date_pattern, s)
        if match:
            date_part = match.group()
            if len(date_part) == 6:
                date_format = '%Y%m'
            elif len(date_part) == 8:
                date_format = '%Y%m%d'
            else:
                raise ValueError(f"无法解析日期部分：{date_part}")

            date_obj = datetime.strptime(date_part, date_format)

            if previous_date:
                delta = date_obj - previous_date
                time_difference_days = delta.days
                time_differences.append(time_difference_days)

            previous_date = date_obj

    # 将时间差保存到CSV文件
    with open(r'E:\XMU\231002_reproduce\preprocess_data\time_differences1002.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["文件1", "文件2", "时间差（天）"])

        for i in range(len(time_differences)):
            if i + 1 < len(date_strings):
                writer.writerow([date_strings[i], date_strings[i + 1], time_differences[i]])