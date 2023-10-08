import numpy as np
from scipy.interpolate import griddata
import netCDF4 as nc
import pandas as pd
from glob import glob
import os
from tqdm import tqdm


def fill_nan_data(data_path, nan_value, fill_method='mean'):
    # 打开原始数据文件以读写方式
    data = nc.Dataset(data_path, 'r+')
    keys = data.variables.keys()
    
    for key in keys:
        if len(data[key][:].shape) == 3:
            original_data = data[key][:].data
            index = np.where(original_data == nan_value)
            original_data[index] = np.nan
            
            if fill_method == 'mean':
                # 使用均值填充缺失值
                original_data[index] = np.nanmean(original_data)
            elif fill_method == 'median':
                # 使用中位数填充缺失值
                original_data[index] = np.nanmedian(original_data)
            elif fill_method == 'nearest':
                pass
                # need to complete
            else:
                raise ValueError('fill_method is not supported')
            data[key][:] = original_data
    
    # 关闭文件以保存修改
    data.close()
           



def get_weather_data(data_path, lon, lat):
    lons = np.linspace(117, 118, 11)
    lats = np.linspace(23, 24, 11)

    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = nc.Dataset(data_path)
    keys = data.variables.keys()
    
    # 用于存储插值后的数据
    interpolated_data = []
    
    for key in keys:
        if len(data[key][:].shape) == 3:
            data1 = data[key][:].data[0]
            data2 = data[key][:].data[1]

            # 使用griddata进行双线性插值
            target_value1 = griddata((lon_grid.flatten(), lat_grid.flatten()), data1.flatten(), (lon, lat), method='linear')
            target_value2 = griddata((lon_grid.flatten(), lat_grid.flatten()), data2.flatten(), (lon, lat), method='linear')
            
            interpolated_data.append((key + '_1', target_value1))
            interpolated_data.append((key + '_2', target_value2))
    
    return interpolated_data


def get_precip_data(year , lon1, lat1, data_path = r'E:\XMU\231002_reproduce\weather_1005\precipitation'):
    lons = [116.25, 118.75]
    lats = [21.25, 23.75, 26.25]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    year = str(year)
    precip_path = sorted(glob(os.path.join(data_path, '*'+year+'*')))
    keys = 'precip'
    interpolated_data = []
    for k, precip in enumerate(precip_path):
        data = nc.Dataset(precip)
        temp_data = data[keys][:].data[0]
        target_value = griddata((lon_grid.flatten(), lat_grid.flatten()), temp_data.flatten(), (lon1, lat1), method='linear')
        interpolated_data.append((keys + f'_{k+1}', target_value))
    return interpolated_data
        






if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\continual_data.xlsx'
    data = pd.read_excel(data_path)
    nc_path = glob(os.path.join(r'E:\XMU\231002_reproduce\weather_1005', '2*', '*.nc'))
    for ncs in nc_path:
        fill_nan_data(ncs, -32767)
    # 新的DataFrame用于存储插值后的数据
    interpolated_data_df = pd.DataFrame()
    
    for i in tqdm(range(len(data))):
        lon, lat = data.loc[i, 'lon'], data.loc[i, 'lat']
        year = data.loc[i, 'year']
        year = str(year)
        nc_path = glob(os.path.join(r'E:\XMU\231002_reproduce\weather_1005', year, '*.nc'))[0]
        
        # 获取插值后的数据
        interpolated_values = get_weather_data(nc_path, lat, lon)
        precip_values = get_precip_data(year, lat, lon)
        
        # 将插值后的数据添加到新的DataFrame中
        for key, value in interpolated_values:
            interpolated_data_df.loc[i, key] = value
        for key, value in precip_values:
            interpolated_data_df.loc[i, key] = value
    # 将新的DataFrame保存到Excel文件
    interpolated_data_df.to_excel(r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data.xlsx', index=False)
    combined_data = pd.concat([data, interpolated_data_df], axis=1)
    # 将新的DataFrame保存到Excel文件
    combined_data.to_excel(r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather.xlsx', index=False)
