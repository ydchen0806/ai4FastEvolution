from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm





def get_data(data_path):
    if os.path.basename(data_path).split('.')[1] == 'xlsx' or os.path.basename(data_path).split('.')[1] == 'xls':
        data = pd.read_excel(data_path)
    elif os.path.basename(data_path).split('.')[1] == 'csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError('data format is not supported')
    return data

def get_cluster(data, regress_data, n_clusters, use_feature, save_dir):
    save_dir = os.path.join(save_dir, f'{n_clusters}_clusters')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    feature_data = regress_data.loc[:, use_feature]
    feature_data = feature_data.dropna(axis=1, how='all')
    feature_data = feature_data.fillna(method='ffill')
    scaler = StandardScaler()
    scaler.fit(feature_data)
    feature_data = scaler.transform(feature_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_data)
    data['cluster'] = kmeans.labels_
    data.to_excel(os.path.join(save_dir, 'cluster.xlsx'), index=False)
    
    # 创建一个Excel写入器
    with pd.ExcelWriter(os.path.join(save_dir, 'cluster_data.xlsx'), engine='openpyxl') as writer:
        # 写入绝对数值到工作表"Count"
        result_count = data.groupby(['year', 'cluster']).size().unstack(fill_value=0)
        result_count.to_excel(writer, sheet_name='Count')

        # 计算每个分类随着年份的占比
        result_percentage = result_count.div(result_count.sum(axis=1), axis=0) * 100

        # 写入占比到工作表"Percentage"
        result_percentage.to_excel(writer, sheet_name='Percentage')

        # 保存聚类中心
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=use_feature)
        cluster_centers.to_excel(writer, sheet_name='Cluster_centers')

        # 保存每个类的平均生长速率
        result_mean = data.groupby('cluster').mean()['growth_rate']
        result_mean.to_excel(writer, sheet_name='Mean_growth_rate')
    
    return data


if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather_dist.xlsx'
    data = get_data(data_path)
    regress_path = r'E:\XMU\231002_reproduce\preprocess_data\pca_nc_data\MGWR\MGWR_session_results_results.csv'
    save_dir = r'E:\XMU\231002_reproduce\preprocess_data'
    regress_data = get_data(regress_path)
    n_clusters = 4
    # use_feature = ['beta_pca_0', 'beta_pca_1', 'beta_pca_2', 'beta_pca_3']
    use_feature = ['beta_pca_0', 'beta_pca_1', 'beta_pca_2']
    # use_feature = ['beta_pca_0',  'beta_pca_2']
    data = get_cluster(data, regress_data, n_clusters, use_feature, save_dir)
    print(data.groupby('cluster').mean()['growth_rate'])
