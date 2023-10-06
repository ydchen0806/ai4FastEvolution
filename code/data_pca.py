import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import StandardScaler


def get_data(data_path):
    if os.path.basename(data_path).split('.')[1] == 'xlsx' or os.path.basename(data_path).split('.')[1] == 'xls':
        data = pd.read_excel(data_path)
    elif os.path.basename(data_path).split('.')[1] == 'csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError('data format is not supported')
    return data

def pca_data(data, if_random = False, n_components = 4, feature_path=None ,start_feature = 'u10_1', end_feature = 'precip_12', save_dir = None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if feature_path:
        feature_data = get_data(feature_path)
        feature_data = feature_data[feature_data['importance'] > 0.0]
        top_feature = feature_data.iloc[:,0].values.tolist()
        
        # 使用 isin() 来筛选出在 top_feature 中的列
        selected_features = data.loc[:, start_feature:end_feature].columns[data.loc[:, start_feature:end_feature].columns.isin(top_feature)]

        # 获取包含筛选出的列的数据
        feature_data = data.loc[:, selected_features]
    else:
        feature_data = data.loc[:, start_feature:end_feature]
        
    feature_data = feature_data.dropna(axis=1, how='all')
    feature_data = feature_data.fillna(method='ffill')
    feature_columns = feature_data.columns
    scaler = StandardScaler()
    scaler.fit(feature_data)
    feature_data = scaler.transform(feature_data)
    random.seed(0)
    if if_random:
        feature_data += np.random.normal(0, 0.01, feature_data.shape)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(feature_data)
    pca_results = pca.transform(feature_data)
    for i in range(n_components):
        data[f'pca_{i}'] = pca_results[:, i]
    # calculate the explained variance
    exp_var = pca.explained_variance_ratio_
    exp_dataframe = pd.DataFrame(exp_var, columns=['explained_variance_ratio'])
    exp_dataframe['cumulative_explained_variance_ratio'] = exp_dataframe['explained_variance_ratio'].cumsum()
    exp_dataframe.index = [f'pca_{i}' for i in range(n_components)]
    if save_dir:
        exp_dataframe.to_excel(os.path.join(save_dir, 'explained_variance.xlsx'))
    # plot the explained variance
    plt.bar(range(n_components), exp_var)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(n_components))
    plt.title('Explained Variance')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'explained_variance.png'))
    plt.show()
    # calculate the importance of each feature
    feature_importance = pca.components_
    feature_importance = pd.DataFrame(feature_importance, columns=feature_columns)
    feature_importance.index = [f'pca_{i}' for i in range(n_components)]
    feature_importance.to_excel(os.path.join(save_dir, 'feature_importance.xlsx'))
    top_10_features = pd.DataFrame(columns=['PCA_Component', 'Top_Features'])

    for i in range(n_components):
        # 获取第i个主成分的特征重要性
        component_importance = feature_importance.iloc[i]

        # 对特征重要性进行降序排序，并选择前10个因子
        top_10 = component_importance.sort_values(ascending=False)[:10]

        # 将结果保存到top_10_features DataFrame中
        top_10_features = top_10_features.append({
            'PCA_Component': f'pca_{i}',
            'Top_Features': ', '.join(top_10.index.tolist())
        }, ignore_index=True)
    if save_dir:
        top_10_features.to_excel(os.path.join(save_dir, 'top_10_features.xlsx'))
    # plot the feature importance
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_importance, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=90)
    plt.yticks(range(n_components), [f'pca_{i}' for i in range(n_components)])
    plt.title('Feature Importance')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.show()
    # save the data

    if save_dir:
        np.save(os.path.join(save_dir, 'data_pca.npy'), data)
        data.to_excel(os.path.join(save_dir, 'data_with_pca.xlsx'))
    return data

if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather_dist.xlsx'
    save_dir = r'E:\XMU\231002_reproduce\preprocess_data\pca_nc_data'
    feature_path = r'E:\XMU\231002_reproduce\code\auto_ml\feature_importance.xlsx'
    data = get_data(data_path)
    data = pca_data(data, feature_path=feature_path,save_dir=save_dir)
    print(data)