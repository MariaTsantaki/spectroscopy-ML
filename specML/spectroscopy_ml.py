from model_training import Data, Model
from simple_minimization import Minimizer
from minimization import minimize_ML
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from utils import int_spectrum_synth, save_and_compare_synthetic
from matplotlib import cm
import argparse
from glob import glob
import random
from astropy.io import fits

def self_check(X_test, y_test, model, clf, plot=True):
    x_pred = []
    for i, y in enumerate(y_test.values[:]):
        minimizer = Minimizer(y, model)
        res = minimizer.minimize()
        x_pred.append([res.x[0], res.x[1], res.x[2], res.x[3]])
    xlabel = ['teff', 'logg', 'feh', 'alpha']
    params = pd.DataFrame(np.array(x_pred), columns=xlabel)

    for i, label in enumerate(xlabel):
        print(label)
        print(np.mean(X_test[label].values - params[label].values))
        print(np.std(X_test[label].values - params[label].values))

    if plot:
        plt.figure()
        plt.scatter(X_test['teff'], X_test['teff'] - params['teff'].values, c=X_test['feh'], alpha=0.8, cmap=cm.jet)
        plt.plot([4000, 6700], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.grid()
        plt.title('Teff')
        plt.show()
        plt.figure()
        plt.scatter(X_test['teff'], X_test['teff'] - params['teff'].values, c=X_test['logg'], alpha=0.8, cmap=cm.jet)
        plt.plot([4000, 6700], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.grid()
        plt.title('Teff')
        plt.show()
        plt.figure()
        plt.scatter(X_test['logg'], X_test['logg'] - params['logg'].values, c=X_test['teff'], alpha=0.8, cmap=cm.jet)
        plt.plot([1, 5], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.title('logg')
        plt.grid()
        plt.show()
        plt.figure()
        plt.scatter(X_test['logg'], X_test['logg'] - params['logg'].values, c=X_test['feh'], alpha=0.8, cmap=cm.jet)
        plt.plot([1, 5], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.grid()
        plt.title('logg')
        plt.show()
        plt.figure()
        plt.scatter(X_test['feh'], X_test['feh'] - params['feh'].values, c=X_test['teff'], alpha=0.8, cmap=cm.jet)
        plt.plot([-2.2, 0.5], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.colorbar()
        plt.grid()
        plt.title('feh')
        plt.show()
        plt.figure()
        plt.scatter(X_test['feh'], X_test['feh'] - params['feh'].values, c=X_test['logg'], alpha=0.8, cmap=cm.jet)
        plt.plot([-2.2, 0.5], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.grid()
        plt.title('feh')
        plt.colorbar()
        plt.show()
        plt.scatter(X_test['alpha'], X_test['alpha'] - params['alpha'].values, c=X_test['teff'], alpha=0.8, cmap=cm.jet)
        plt.plot([-0.4, 0.5], [0, 0], color='k', linestyle='-', linewidth=2)
        plt.grid()
        plt.title('alpha')
        plt.colorbar()
        plt.show()

def test_set_synth(model, continuum=None):

    #read synthetic fluxes
    path_of_grid = 'data/results/'
    spec = glob(path_of_grid + '*.spec')
    params = []
    for s in spec[:]:
        y, w = int_spectrum_synth(s, continuum)
        minimizer = Minimizer(y, model)
        res = minimizer.minimize()
        params.append([res.x[0], res.x[1], res.x[2], res.x[3]])

    xlabel = ['teff', 'logg', 'feh', 'alpha']
    params = np.array(params)
    d = [spec, params[:, 0], params[:, 1], params[:, 2], params[:, 3]]
    d = np.array(d)
    spec = list(map(lambda x: x.split('/')[-1], spec))
    d = {'specname': spec, 'teff': params[:, 0], 'logg': params[:, 1], 'metal': params[:, 2], 'alpha': params[:, 3]}

    results = save_and_compare_synthetic(d, class_name='linear')
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectroscopic parameters with ML')
    parser.add_argument('-c', '--classifier',
                        help='Which classifier to use',
                        choices=('linear', 'ridge', 'lasso', 'ridgeCV', 'lassolars'),
                        default='linear')
    args = parser.parse_args()
    clf = args.classifier

    d = Data('spec_ML.hdf', with_quadratic_terms=True, split=True, scale=True)
    d.flux_removal(cutoff=0.999, percent=20)
    c = d.get_wavelength()

    model = Model(d, classifier=clf, save=True, load=False)
    parser = argparse.ArgumentParser(description='Spectroscopic parameters with ML')
    parser.add_argument('-c', '--classifier',
                        help='Which classifier to use',
                        choices=('linear', 'ridge', 'lasso', 'ridgeCV', 'lassolars'),
                        default='linear')
    args = parser.parse_args()
    clf = args.classifier

    d = Data('spec_ML.hdf', with_quadratic_terms=True, split=True, scale=True)
    d.flux_removal(cutoff=0.999, percent=20)
    c = d.get_wavelength()
    model = Model(d, classifier=clf, save=True, load=False)
    X_test = d.X_test
    y_test = d.y_test


    self_check(X_test, y_test, model, clf, plot=True)
    test_set_synth(model, continuum=c)
