from __future__ import division
import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
try:
    import cPickle
except ImportError:
    import _pickle as cPickle

from utils import create_combined, prepare_linelist, prepare_spectrum, save_and_compare_apogee, save_and_compare_synthetic


def poly_clf():
    polynomial_features = PolynomialFeatures(degree=3, include_bias=False)
    linear_regression = linear_model.LinearRegression()
    clf = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    #clf.fit(X[:, np.newaxis], y)
    #y_pred = clf.predict(X_test[:, np.newaxis])
    return clf


def validation():

    if not os.path.isfile('combined_spec.csv'):
        create_combined()

    df = pd.read_csv('combined_spec.csv', index_col=0)
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[:-7]
    ylabel = df.columns.values[-7:-1]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]

    param_range = np.linspace(0.0001, 0.01, 5)

    train_scores, test_scores = validation_curve(linear_model.Lasso(), X, y, param_name="alpha", param_range=param_range, scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel("alpha")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def train(clf, mod, save=True, plot=True, scale=True):

    if not os.path.isfile('combined_spec.csv'):
        create_combined()

    df = pd.read_csv('combined_spec.csv', index_col=0)
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[:-7]
    ylabel = df.columns.values[-7:-1]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]

    if scale:
        #Is this ok?
        X = preprocessing.robust_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clffit = clf.fit(X_train, y_train)
    print(clffit)
    N = len(y_test)
    t = time()
    y_pred = clf.predict(X_test)
    t = time()-t
    speedup = 60*N/t
    print('Calculated parameters for {} stars in {:.2f}ms'.format(N, t*1e3))
    print('Speedup: {} million times'.format(int(speedup/1e6)))
    print('Test set score: {:.2f}'.format(clf.score(X_test, y_test)))
    print('Training set score: {:.2f}'.format(clffit.score(X_train, y_train)))

    for i, label in enumerate(ylabel):
        score = mean_absolute_error(y_test[label], y_pred[:, i])
        print('Mean absolute error for {}: {:.2f}'.format(label, score))
        if plot:
            plt.figure()
            plt.plot(y_test[label], y_test[label].values - y_pred[:, i], 'o')
            plt.grid()
            plt.title(label)
            plt.savefig(label + '_' + mod + '.png')
            plt.show()

    if save:
        with open('FASMA_ML.pkl', 'wb') as f:
            cPickle.dump(clf, f)
    return clf


def train_models(mod):

    print('Selected model: %s' % mod)
    if mod == 'linear':
        clf = linear_model.LinearRegression(n_jobs=-1)
    elif mod == 'lasso':
        clf = linear_model.Lasso(alpha=0.001, max_iter=5000, normalize=True)
    elif mod == 'lassolars':
        clf = linear_model.LassoLars(alpha=0.001)
    elif mod == 'multilasso':
        clf = linear_model.MultiTaskLasso(alpha=0.1)
    elif mod == 'ridgeCV':
        clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    elif mod == 'ridge':
        clf = linear_model.Ridge(alpha=[0.01])
    elif mod == 'bayes':
        clf = linear_model.BayesianRidge()
    elif mod == 'huber':
        clf = linear_model.HuberRegressor()
    elif mod == 'poly':
        clf = poly_clf()

    clf = train(clf, mod, save=True, plot=True)
    return clf


def test_set(clf, model, scale=True):

    spec = np.genfromtxt('obs_synth.lst', dtype='str')
    params = []
    for s in spec:

        x = prepare_spectrum(s)
        if scale:
            x = preprocessing.robust_scale(x)

        p = clf.predict(x)[0]
        params.append(p)
        #print('Star: %s' % s)
        #print('\nStellar atmospheric parameters:')
        #print('Teff:   {:.0f} K'.format(p[0]))
        #print('logg:   {:.2f} dex'.format(p[1]))
        #print('[M/H]:  {:.2f} dex'.format(p[2]))
        #print('alpha:  {:.2f} dex'.format(p[3]))
        #print('vt:     {:.2f} km/s'.format(p[4]))
        #print('vmac:   {:.2f} km/s'.format(p[5]))
        #print('vsini:  {:.2f} km/s'.format(p[6]))
    params = np.array(params)
    d = [spec, params[:, 0], params[:, 1], params[:, 2], params[:, 3]]
    d = np.array(d)
    spec = list(map(lambda x: x.split('/')[-1], spec))
    d = {'specname': spec, 'teff': params[:, 0], 'logg': params[:, 1], '[M/H]': params[:, 2], 'alpha': params[:, 3]}

    save_and_compare_synthetic(d, model)
    #save_and_compare_apogee(d)
    return


def lasso(alpha):
    clf = linear_model.Lasso(alpha=alpha, max_iter=10000, normalize=True)
    mod = 'lasso_' + str(alpha)
    train(clf, mod)
    return


def ridge(alpha):
    return

if __name__ == '__main__':

    models = ['linear']
    #models = ['linear', 'lasso', 'multilasso', 'lassolars', 'ridge', 'ridgeCV', 'bayes', 'huber', 'poly']
    validation()
    #for mod in models:
    #    clf = train_models(mod)
    #    test_set(clf, mod)

    #alpha = [0.00001, 0.0001, 0.001]
    #for a in alpha:
    #    lasso(a)