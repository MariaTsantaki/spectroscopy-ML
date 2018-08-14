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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import validation_curve
from minimization import minimize_ML
from random import randint
try:
    import cPickle
except ImportError:
    import _pickle as cPickle

from utils import create_combined, prepare_spectrum, save_and_compare_apogee, save_and_compare_synthetic, prepare_spectrum_synth
from matplotlib import cm

def poly_clf():
    polynomial_features = PolynomialFeatures(degree=2, interaction_only=True)
    linear_regression = linear_model.LinearRegression()
    clf = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    return clf


def train(clf, model, save=True, cutoff=0.99, percent=50, plot=True, scale=False):
    # Model just for saving options
    if not os.path.isfile('combined_spec.csv'):
        create_combined()

    df = pd.read_csv('spec_ML.csv', index_col=0)
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[-7:-3]
    ylabel = df.columns.values[:-7]
    X = df.loc[:, xlabel]  #Parameters
    X['teff**2']   = X['teff'] ** 2
    X['logg**2']   = X['logg'] ** 2
    X['feh**2']    = X['feh'] ** 2
    X['teff*logg'] = X['teff'] * X['logg']
    X['teff*feh']  = X['teff'] * X['feh']
    X['logg*feh']  = X['logg'] * X['feh']
    y = df.loc[:, ylabel] #Fluxes

    # select continuum
    continuum = []
    for ylab in ylabel[:]:
        flux = y[ylab]
        flux_cont = flux.loc[flux > cutoff]
        if (len(flux_cont)/len(flux))*100 > percent:
            continuum.append(ylab)

    columns = np.array(continuum)
    y.drop(columns, inplace=True, axis=1)
    print('The percentage of flux points dropped is %s with a %s cutoff.' % (percent, cutoff))
    print('The number of flux points is %s from the original %s.' % (len(ylabel)-len(continuum), len(ylabel)))

    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    # Training of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    clf = clf.fit(X_train, y_train)

    # Make predictions
    t = time()
    x_pred = []
    for i, y in enumerate(y_test.values[:]):
        p = minimize_ML(clf, y, scale=scale)
        x_pred.append(p)
        print(X_test.values[i])
    params = pd.DataFrame(np.array(x_pred), columns=xlabel)
    print(params)
    t = time()-t

    print('Calculated parameters for {} stars in {:.2f}ms'.format(len(y_test), t*1e3))
    print('Test set score: {:.2f}'.format(clf.score(X_test, y_test)))

    for i, label in enumerate(xlabel):
        #score = mean_absolute_error(y_test[label], y_pred[:, i])
        #print('Mean absolute error for {}: {:.2f}'.format(label, score))
        if plot:
            plt.figure()
            plt.scatter(X_test[label], X_test[label].values - params[label].values, s=70, alpha=0.4)
            plt.grid()
            plt.title(label)
            #plt.savefig(label + '_' + model + '.png')
            plt.show()

    if save:
        with open('FASMA_ML.pkl', 'wb') as f:
            cPickle.dump(clf, f)
    print('Selected model: %s' % clf)
    return clf, continuum


def train_models(mod, save=True, cutoff=0.999, percent=50, plot=True, scale=False):

    if mod == 'linear':
        clf = linear_model.LinearRegression(n_jobs=-1)
    elif mod == 'lasso':
        clf = linear_model.Lasso(alpha=1000, max_iter=10000, tol=0.001, normalize=True, positive=True)
    elif mod == 'lassolars':
        clf = linear_model.LassoLars(alpha=0.001)
    elif mod == 'multilasso':
        clf = linear_model.MultiTaskLasso(alpha=0.1)
    elif mod == 'ridgeCV':
        clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    elif mod == 'ridge':
        clf = linear_model.Ridge(alpha=[1000])
    elif mod == 'bayes':
        clf = linear_model.BayesianRidge()
    elif mod == 'huber':
        clf = linear_model.HuberRegressor()
    elif mod == 'poly':
        #clf = poly_clf()
        clf = PolynomialFeatures(degree=2)

    clf, continuum = train(clf, mod, save=save, cutoff=cutoff, percent=percent, plot=plot, scale=scale)
    return clf, continuum


def test_set(clf, model, continuum=None, fname='obs_synth.lst', scale=False, mode='synth'):

    #here model is just for saving the plot files
    spec = np.genfromtxt(fname, dtype='str')
    params = []
    if mode == 'synth':
        for s in spec[:]:
            y, w = prepare_spectrum_synth(s, continuum)
            p = minimize_ML(clf, y[0], scale=False)
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

            #f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
            #ax1.plot(w, x[0])
            #ax2.scatter(w, clf.coef_[0])
            #ax3.scatter(w, clf.coef_[1])
            #ax4.scatter(w, clf.coef_[2])
            #ax5.scatter(w, clf.coef_[3])
            #f.subplots_adjust(hspace=0)
            #plt.grid(True)
            #plt.show()

        params = np.array(params)
        d = [spec, params[:, 0], params[:, 1], params[:, 2], params[:, 3]]
        d = np.array(d)
        spec = list(map(lambda x: x.split('/')[-1], spec))
        d = {'specname': spec, 'teff': params[:, 0], 'logg': params[:, 1], 'metal': params[:, 2], 'alpha': params[:, 3]}
        results = save_and_compare_synthetic(d, model)

    elif mode == 'apogee':
        for s in spec[:]:
            y, w = prepare_spectrum(s, continuum)
            p = minimize_ML(clf, y[0], scale=False)
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

            #f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
            #ax1.plot(w, x[0])
            #ax2.scatter(w, clf.coef_[0])
            #ax3.scatter(w, clf.coef_[1])
            #ax4.scatter(w, clf.coef_[2])
            #f.subplots_adjust(hspace=0)
            #plt.grid(True)
            #plt.show()

        params = np.array(params)
        d = [spec, params[:, 0], params[:, 1], params[:, 2], params[:, 3]]
        d = np.array(d)
        spec = list(map(lambda x: x.split('/')[-1], spec))
        d = {'specname': spec, 'teff': params[:, 0], 'logg': params[:, 1], 'metal': params[:, 2], 'alpha': params[:, 3]}
        results = save_and_compare_apogee(d, model)

        #teff = np.dot(clf.coef_[0], x[0]) + clf.intercept_[0]
        #print('teff', teff)
    else:
        print('What are your data?')
        results = []
    return results


def ridge_all(alpha, cutoff=0.999, percent=50, fname_synth='obs_synth300.lst', fname_obs='obs.lst'):
    clf = linear_model.Ridge(alpha=[alpha])
    #clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0])
    model = 'ridge_' + str(alpha) + '_' + str(percent) + '_' + str(cutoff)
    clf, continuum = train(clf, model, save=True, cutoff=cutoff, percent=percent, plot=False, scale=False)
    #results_synth  = test_set(clf, model, continuum, fname=fname_synth, mode='synth')
    results_apogee = test_set(clf, model, continuum, fname=fname_obs, mode='apogee')

    #print(results_synth)
    #print(results_apogee)
    return


if __name__ == '__main__':


    #models = ['linear', 'lasso', 'multilasso', 'lassolars', 'ridge', 'ridgeCV', 'bayes', 'huber', 'poly']
    models = ['linear']
    #validation()
    for mod in models:
        clf, continuum = train_models(mod, save=True, cutoff=0.995, percent=40, plot=True, scale=True)
        #with open('FASMA_ML.pkl', 'rb') as f:
        #    clf = cPickle.load(f)
        #print(clf)
        #results = test_set(clf, mod, continuum=continuum, fname='obs_synth300.lst', scale=False, mode='synth')
        #results = test_set(clf, mod, continuum=continuum, fname='obs.lst', scale=False, mode='apogee')

    #r = ridge_all(1000, cutoff=0.999, percent=40, fname_synth='obs_synth300.lst', fname_obs='obs.lst')
    #r = ridge(0.1, cutoff=0.999, percent=40, fname='obs.lst')
    #alpha = [900, 950, 1000, 1050, 1100]
    #for a in alpha:
    #    r = ridge_all(a, cutoff=0.999, percent=40, fname='obs_synth300.lst')
