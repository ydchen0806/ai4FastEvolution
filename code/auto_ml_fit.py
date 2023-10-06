import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from autogluon.tabular import TabularDataset, TabularPredictor
import random
from matplotlib import pyplot as plt
import seaborn as sns


def get_data(data_path):
    if os.path.basename(data_path).split('.')[1] == 'xlsx' or os.path.basename(data_path).split('.')[1] == 'xls':
        data = pd.read_excel(data_path)
    elif os.path.basename(data_path).split('.')[1] == 'csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError('data format is not supported')
    return data

def auto_fit(data, y_label, x_label, save_path):
    data = data[[y_label] +  x_label]
    data = data.dropna()
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    data = data.reset_index(drop=True)
    # data.index = random.sample(data.index.tolist(), len(data))
    train_data = TabularDataset(data)
    predictor = TabularPredictor(label=y_label, path=save_path).fit(train_data)
    truth_y = train_data[y_label]
    y_pred = predictor.predict(train_data)
    mse = predictor.evaluate_predictions(y_true=truth_y, y_pred=y_pred, auxiliary_metrics=True)
    diff_y = truth_y - y_pred
    df = pd.DataFrame({'truth': truth_y, 'predict': y_pred, 'diff': diff_y})
    df.to_excel(os.path.join(save_path, 'truth_vs_predict.xlsx'))
    print(f'The mse of {x_label} is {mse}')
    plt.figure(figsize=(10, 10))
    plt.plot(truth_y, label='Truth', color='blue')
    plt.plot(y_pred, label='Predict', color='red')
    plt.xlabel('Samples')
    plt.ylabel('Growth Rate')
    plt.title('Truth vs Predict')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'truth_vs_predict.png'))
    plt.show()
    return predictor, train_data

def auto_predict(predictor, data, y_label, x_label, save_path):
    data = data[[y_label, x_label]]
    data = data.dropna()
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    data = data.reset_index(drop=True)
    test_data = TabularDataset(data)
    y_pred = predictor.predict(test_data)
    return y_pred

def auto_fit_predict(data, y_label, x_label, save_path):
    from sklearn.metrics import mean_squared_error
    # split the data into train and test random
    data = data[[y_label, x_label]]
    data = data.dropna()
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    data = data.reset_index(drop=True)
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)
    predictor = TabularPredictor(label=y_label, path=save_path).fit(train_data)
    y_pred = predictor.predict(test_data)
    y_true = test_data[y_label]
    mse = mean_squared_error(y_true, y_pred)
    print(f'The mse of {x_label} is {mse}')
    return mse

def get_importance(predictor, x_train, save_path):
    feature_importance = predictor.feature_importance(x_train)
    top_10_features = feature_importance.head(10)  # 获取前十个重要特征
    top_10_features.to_excel(os.path.join(save_path, 'top_10_features.xlsx'))
    feature_importance.to_excel(os.path.join(save_path, 'feature_importance.xlsx'))
    return feature_importance

if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\interpolated_data_with_weather_dist.xlsx'
    data = pd.read_excel(data_path)
    save_path = r'E:\XMU\231002_reproduce\code\auto_ml'
    os.makedirs(save_path, exist_ok=True)
    y_label = 'growth_rate'
    x_label = ['u10_1', 'u10_2', 'v10_1', 'v10_2', 'd2m_1', 'd2m_2', 't2m_1', 't2m_2',
       'evabs_1', 'evabs_2', 'evaow_1', 'evaow_2', 'evatc_1', 'evatc_2',
       'evavt_1', 'evavt_2', 'fal_1', 'fal_2', 'lblt_1', 'lblt_2', 'licd_1',
       'licd_2', 'lict_1', 'lict_2', 'lmld_1', 'lmld_2', 'lmlt_1', 'lmlt_2',
       'lshf_1', 'lshf_2', 'ltlt_1', 'ltlt_2', 'lai_hv_1', 'lai_hv_2',
       'lai_lv_1', 'lai_lv_2', 'pev_1', 'pev_2', 'ro_1', 'ro_2', 'src_1',
       'src_2', 'skt_1', 'skt_2', 'es_1', 'es_2', 'stl1_1', 'stl1_2', 'stl2_1',
       'stl2_2', 'stl3_1', 'stl3_2', 'stl4_1', 'stl4_2', 'ssro_1', 'ssro_2',
       'slhf_1', 'slhf_2', 'ssr_1', 'ssr_2', 'str_1', 'str_2', 'sp_1', 'sp_2',
       'sro_1', 'sro_2', 'sshf_1', 'sshf_2', 'ssrd_1', 'ssrd_2', 'strd_1',
       'strd_2', 'e_1', 'e_2', 'tp_1', 'tp_2', 'swvl1_1', 'swvl1_2', 'swvl2_1',
       'swvl2_2', 'swvl3_1', 'swvl3_2', 'swvl4_1', 'swvl4_2', 'precip_1',
       'precip_2', 'precip_3', 'precip_4', 'precip_5', 'precip_6', 'precip_7',
       'precip_8', 'precip_9', 'precip_10', 'precip_11', 'precip_12','size','pos_x',
       'pos_y','min_distance','year']
    predictor, train_data = auto_fit(data, y_label, x_label, save_path)
    # y_pred = auto_predict(predictor, data, y_label, x_label, save_path)
    # mse = auto_fit_predict(data, y_label, x_label, save_path)
    feature_importance = get_importance(predictor, train_data, save_path)
    print(feature_importance)
    # print(y_pred)
    # print(mse)
    # print(predictor.feature_importance(x_label))
    # print(predictor.leaderboard())
    # print(predictor.feature_importance(x_label))