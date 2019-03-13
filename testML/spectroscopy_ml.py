from model_training_poly import Data, Model
from minimization import minimize_ML
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from utils import save_and_compare_synthetic, save_and_compare_sn4, meanstdv
from matplotlib import cm
import argparse
from glob import glob
import random
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import sklearn.metrics

def test_set_synth(model, clf=None, continuum=None):

    #read synthetic fluxes
    data = Data('spec_ML_kurucz_validation.hdf', split=False, scale=False, feature=False)
    X_para = data.X
    y_para = data.y
    if continuum is not None:
        y_para.drop(continuum, inplace=True, axis=1)
    y_test = y_para.values

    params = []
    for i, y in enumerate(y_test[:]):
        minimizer = minimize_ML(y, model)
        p = minimizer.minimize()
        params.append([p[0], p[1], p[2], p[3]])

    params = np.array(params)
    d = [X_para.values[:, 0], X_para.values[:, 1], X_para.values[:, 2], X_para.values[:, 3], params[:, 0], params[:, 1], params[:, 2], params[:, 3]]
    d = np.array(d)
    save_and_compare_synthetic(d)
    return

def test_set_sn4(model, clf=None, continuum=None):

    #read synthetic fluxes
    data = Data('spec_ML_sn4.hdf', split=False, scale=False, feature=False)
    X_para = data.X
    y_para = data.y
    if continuum is not None:
        y_para.drop(continuum, inplace=True, axis=1)
    y_test = y_para.values

    params = []
    for i, y in enumerate(y_test[:]):
        y[y > 1.0] = 1.0
        minimizer = minimize_ML(y, model)
        p = minimizer.minimize()
        params.append([p[0], p[1], p[2], p[3]])

    params = np.array(params)
    d = [X_para.values[:, 0], X_para.values[:, 1], X_para.values[:, 2], params[:, 0], params[:, 1], params[:, 2]]
    d = np.array(d)
    save_and_compare_sn4(d)
    return


def model_evaluation(d, clf=None, continuum=None):

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    data = Data('spec_ML_kurucz_validation.hdf', split=False, scale=True, feature=False)
    X_test = data.Xml
    print(X_test)
    y_para = data.y
    if continuum is not None:
        y_para.drop(continuum, inplace=True, axis=1)
    y_test = y_para.values

    mae_a = []
    mse_a = []
    r2_a  = []
    for alpha in alphas:
        model = Model(d, classifier=clf, alpha=alpha, save=True, load=False)
        y_pred = model.predict_output(X_test)
        scores = cross_val_score(model.clf, X_test, y_test, cv=3)
        print(scores)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        r2  = sklearn.metrics.r2_score(y_test, y_pred)
        mae_a.append(mae)
        mse_a.append(mse)
        r2_a.append(r2)
        print('mae: %s, mse: %s, r2: %s' % (round(mae, 4), round(mse, 4), round(r2, 4)))
        test_set_synth(model, clf=clf, continuum=continuum)
        test_set_sn4(model, clf=clf, continuum=continuum)
    plt.plot(alphas, mae_a, 'o', label='mae')
    plt.plot(alphas, mse_a, 'o', label='mse')
    plt.plot(alphas, r2_a,  'o', label='r2')
    plt.legend()
    plt.show()
    for i, x in enumerate(alphas):
        print(x, mae_a[i], mse_a[i], r2_a[i])
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectroscopic parameters with ML')
    parser.add_argument('-c', '--classifier',
                        help='Which classifier to use',
                        choices=('linear', 'ridge', 'lasso',  'lassocv', 'ridgeCV', 'lassolars', 'multilasso', 'tree', 'poly'), default='linear')
    args = parser.parse_args()
    clf = args.classifier

    d = Data('spec_ML_kurucz.hdf', split=False, scale=True, feature=False)
    c = d.flux_removal(cutoff=0.998, percent=30)
    model = Model(d, classifier=clf, save=True, load=False)
    #c = None
    #model_evaluation(d, clf=clf, continuum=c)
    test_set_synth(model, clf=clf, continuum=c)
    test_set_sn4(model, clf=clf, continuum=c)
