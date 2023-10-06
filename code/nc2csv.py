import netCDF4 as nc
import pandas as pd
from glob import glob
import os

def read_nc(data_path, save_dir = None):
    '''
    读取nc文件
    :param data_path: nc文件路径
    :param save_dir: 保存路径
    :return: dataframe
    '''
    data = nc.Dataset(data_path)
    keys = data.variables.keys()
    data_pd = pd.DataFrame()
    for key in keys:
        if key != 'time':
           
            a = data.variables[key][:]
            print(data.variables[key][:].shape)
        # data_pd[key] = data.variables[key][:]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        data_pd.to_csv(os.path.join(save_dir, os.path.basename(data_path).replace('.nc', '.csv')), index=False)
    return data_pd


if __name__ == '__main__':
    father_path = r'E:\XMU\231002_reproduce\weather_data'
    save_dir = r'E:\XMU\231002_reproduce\weather_data_csv'
    data_path = sorted(glob(os.path.join(father_path,'2*er','*.nc')))
    for file in data_path:
        read_nc(file, save_dir)
        print(f'转换{file}完成')