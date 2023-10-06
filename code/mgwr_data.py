import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import compare_surfaces, truncate_colormap
from mgwr.gwr import MGWRResults
from mgwr.gwr import get_AICc, get_AIC, get_BIC
from spglm.family import Gaussian, Binomial, Poisson



def read_data(data_path):
    if os.path.basename(data_path).split('.')[1] == 'xlsx' or os.path.basename(data_path).split('.')[1] == 'xls':
        data = pd.read_excel(data_path)
    elif os.path.basename(data_path).split('.')[1] == 'csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError('data format is not supported')
    return data

def run_mgwr(x, y, coords, bw, selector, family, offset=None, sigma2_v1=True, kernel='bisquare', fixed=False, constant=True, spherical=False, hat_matrix=False):
    if selector:
        selector = Sel_BW(coords, y, x, multi=True)
        bw = selector.search(multi_bw_min=[2], multi_bw_max=[159])
    model = MGWR(coords, y, x, bw, family=family, sigma2_v1=sigma2_v1, kernel=kernel, fixed=fixed, constant=constant, spherical=spherical, hat_matrix=hat_matrix)
    results = model.fit()
    return results


def plot_mgwr_results(results, save_dir=None):
    # plot the coefficients
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(results.params, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Coefficients')
    # plot the standard errors
    plt.subplot(222)
    plt.imshow(results.bse, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Standard Errors')
    # plot the t values
    plt.subplot(223)
    plt.imshow(results.tvalues, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('t values')
    # plot the p values
    plt.subplot(224)
    plt.imshow(results.pvalues, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('p values')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_results.png'))
    plt.show()
    # plot the adjusted R2
    plt.figure(figsize=(10, 10))
    plt.imshow(results.adj_alpha, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Adjusted R2')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_adj_alpha.png'))
    plt.show()
    # plot the AICc
    plt.figure(figsize=(10, 10))
    plt.imshow(results.aicc, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('AICc')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_aicc.png'))
    plt.show()
    # plot the AIC
    plt.figure(figsize=(10, 10))
    plt.imshow(results.aic, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('AIC')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_aic.png'))
    plt.show()
    # plot the BIC
    plt.figure(figsize=(10, 10))
    plt.imshow(results.bic, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('BIC')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_bic.png'))
    plt.show()
    # plot the CV
    plt.figure(figsize=(10, 10))
    plt.imshow(results.cv, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.title('CV')

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_cv.png'))
    plt.show()
    # plot the residuals
    plt.figure(figsize=(10, 10))
    plt.imshow(results.resid_response, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Residuals')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_resid_response.png'))
    plt.show()
    # plot the fitted values
    plt.figure(figsize=(10, 10))
    plt.imshow(results.fitted_values, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Fitted Values')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'mgwr_fitted_values.png'))
    plt.show()

if __name__ == '__main__':
    data_path = r'E:\XMU\231002_reproduce\preprocess_data\pca\data_with_pca.xlsx'
    save_dir = r'E:\XMU\231002_reproduce\preprocess_data\mgwr'
    data = read_data(data_path)
    # get the coordinates
    coords = data[['pos_x', 'pos_y']].values
    # get the dependent variable
    y = data['growth_rate'].values
    # get the independent variables
    x = data[['pca_0', 'pca_1', 'pca_2', 'pca_3']].values
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    # get the bandwidth
    bw = 1000
    # get the family
    family = Gaussian()
    # get the selector
    selector = True
    # get the results
    results = run_mgwr(x, y, coords, bw, selector, family)
    # plot the results
    plot_mgwr_results(results, save_dir=save_dir)
    # get the summary
    print(results.summary())
    # get the coefficients
    print(results.params)
    # get the standard errors
    print(results.bse)
    # get the t values
    print(results.tvalues)
    # get the p values
    print(results.pvalues)
    # get the adjusted R2
    print(results.adj_alpha)
    # get the AICc
    print(results.aicc)
    # get the AIC
    print(results.aic)
    # get the BIC
    print(results.bic)
    # get the CV
    print(results.cv)
    # get the residuals
    print(results.resid_response)
    # get the fitted values
    print(results.fitted_values)
