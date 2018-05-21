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

from utils import create_combined, prepare_linelist, prepare_spectrum, save_and_compare_apogee, save_and_compare_synthetic, prepare_spectrum_synth
from matplotlib import cm

def getData(cutoff=0.9999, percent=50):
    df = pd.read_csv('combined_spec.hdf', index_col=0)
    df.set_index('spectrum', inplace=True)
    ylabel = df.columns.values[:-7]
    xlabel = df.columns.values[-7:]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]

    # select continuum
    continuum = []
    for ylab in ylabel[:]:
        flux = y[ylab]
        flux_cont = flux.loc[flux > cutoff]
        if (len(flux_cont)/len(flux))*100 > percent:
            continuum.append(ylab)

    columns = np.array(continuum)
    y.drop(columns, inplace=True, axis=1)
    return X, y


def x2(y_ML, y_synth):
    err = 1
    chi2 = (y_ML-y_synth)**2/err
    return chi2


def poly_clf():
    polynomial_features = PolynomialFeatures(degree=2, interaction_only=True)
    xpoly = polynomial_features.fit_transform(X)
    linear_regression = linear_model.LinearRegression()
    clf = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    #clf.fit(X[:, np.newaxis], y)
    #y_pred = clf.predict(X_test[:, np.newaxis])
    return clf


def train(clf, model, save=True, cutoff=0.99, percent=50, plot=True, scale=False):
    # Model just for saving options
    if not os.path.isfile('combined_spec.csv'):
        create_combined()

    df = pd.read_csv('spec_ML.csv', index_col=0)
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[-7:-4]
    ylabel = df.columns.values[:-7]
    X = df.loc[:, xlabel]  #Parameters
    #X['teff**2']   = X['teff'] ** 2
    #X['logg**2']   = X['logg'] ** 2
    #X['feh**2']    = X['feh'] ** 2
    #X['teff*logg'] = X['teff'] * X['logg']
    #X['teff*feh']  = X['teff'] * X['feh']
    #X['logg*feh']  = X['logg'] * X['feh']
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
    # Training of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    if scale:
        #scaler = preprocessing.StandardScaler().fit(X_train)
        scaler = preprocessing.RobustScaler()
        y_train = scaler.fit_transform(y_train)
        y_train = pd.DataFrame(y_train, columns=ylabel)
    clf = clf.fit(X_train, y_train)
    print('Selected model: %s' % clf)
    N = len(y_test)
    t = time()
    x_pred = []
    for y in y_test.values[:]:
        p = minimize_ML(clf, y)
        x_pred.append(p)

    params = pd.DataFrame(np.array(x_pred), columns = xlabel)
    t = time()-t
    speedup = 60*N/t
    print('Calculated parameters for {} stars in {:.2f}ms'.format(N, t*1e3))
    #print('Speedup: {} million times'.format(int(speedup/1e6)))
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
    return clf, continuum


def train_models(mod, save=True, cutoff=0.999, percent=50, plot=True, scale=False):

    if mod == 'linear':
        clf = linear_model.LinearRegression(n_jobs=-1)
    elif mod == 'lasso':
        clf = linear_model.Lasso(alpha=0.001, max_iter=10000, tol=0.001, normalize=True, positive=True)
    elif mod == 'lassolars':
        clf = linear_model.LassoLars(alpha=0.001)
    elif mod == 'multilasso':
        clf = linear_model.MultiTaskLasso(alpha=0.1)
    elif mod == 'ridgeCV':
        clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    elif mod == 'ridge':
        clf = linear_model.Ridge(alpha=[0.01])
    elif mod == 'bayes':
        clf = linear_model.BayesianRidge()
    elif mod == 'huber':
        clf = linear_model.HuberRegressor()
    elif mod == 'poly':
        #clf = poly_clf()
        clf = PolynomialFeatures(degree=2)

    clf, continuum = train(clf, mod, save=True, cutoff=0.999, percent=50, plot=True, scale=False)
    return clf, continuum


def test_set(clf, model, continuum=None, fname='obs_synth.lst', mode='synth'):

    #here model is just for saving the plot files
    spec = np.genfromtxt(fname, dtype='str')
    params = []
    if mode == 'synth':
        for s in spec[:]:
            x, w = prepare_spectrum_synth(s, continuum)

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
            x, w = prepare_spectrum(s, continuum)
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


def lasso(alpha):
    clf = linear_model.Lasso(alpha=alpha, max_iter=10000, normalize=True)
    mod = 'lasso_' + str(alpha)
    train(clf, mod)
    return


def ridge(alpha, cutoff=0.999, percent=50, fname='obs_synth.lst'):
    clf = linear_model.Ridge(alpha=[alpha])
    #clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0])
    model = 'ridge_' + str(alpha) + '_' + str(percent)
    clf, continuum = train(clf, model, save=True, cutoff=cutoff, percent=percent, plot=False, scale=False)
    #results = test_set(clf, model, continuum, fname=fname, mode='synth')
    #results = test_set(clf, model, continuum, fname='obs.lst', mode='apogee')
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


def ridge_snrtest():

    snrfile = ['obs_synth100.lst', 'obs_synth200.lst', 'obs_synth300.lst', 'obs_synth400.lst', 'obs_synth500.lst']
    results = []
    #for a in alpha:
    for snr in snrfile:
        r = ridge(0.1, cutoff=0.9999, percent=50, fname=snr)
        results.append(r)
    results = np.array(results)
    r = results.reshape(len(results), 16)
    np.savetxt('results_ML.dat', r, fmt='%s', delimiter='\t')

    label = ['teff', 'logg', 'metal', 'alpha']
    x = [100, 200, 300, 400, 500]
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([50, 550])
    plt.errorbar(x, r[:, 0], yerr=r[:, 2], fmt='o', alpha=0.5, color='green', label='mean')
    plt.errorbar(x, r[:, 1], yerr=r[:, 3], fmt='o', alpha=0.5, color='blue', label='median')
    plt.xlabel(r'SNR')
    plt.legend(frameon=False, numpoints=1)
    plt.grid(True)
    #plt.savefig('teff_ridge_0.1_50_snr.png')
    plt.show()

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([50, 550])
    plt.errorbar(x, r[:, 4], yerr=r[:, 6], fmt='o', alpha=0.5, color='green', label='mean')
    plt.errorbar(x, r[:, 5], yerr=r[:, 7], fmt='o', alpha=0.5, color='blue', label='median')
    plt.xlabel(r'SNR')
    plt.legend(frameon=False, numpoints=1)
    plt.grid(True)
    #plt.savefig('logg_ridge_0.1_50_snr.png')
    plt.show()

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([50, 550])
    plt.errorbar(x, r[:, 8], yerr=r[:, 10], fmt='o', alpha=0.5, color='green', label='mean')
    plt.errorbar(x, r[:, 9], yerr=r[:, 11], fmt='o', alpha=0.5, color='blue', label='median')
    plt.xlabel(r'SNR')
    plt.legend(frameon=False, numpoints=1)
    plt.grid(True)
    #plt.savefig('metal_ridge_0.1_50_snr.png')
    plt.show()

    plt.figure()
    axes = plt.gca()
    axes.set_xlim([50, 550])
    plt.errorbar(x, r[:, 12], yerr=r[:, 14], fmt='o', alpha=0.5, color='green', label='mean')
    plt.errorbar(x, r[:, 13], yerr=r[:, 15], fmt='o', alpha=0.5, color='blue', label='median')
    plt.xlabel(r'SNR')
    plt.legend(frameon=False, numpoints=1)
    plt.grid(True)
    #plt.savefig('alpha_ridge_0.1_50_snr.png')
    plt.show()


def getData():
    df = pd.read_csv('combined_spec.csv', index_col=0)
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[:-7]
    ylabel = df.columns.values[-7:]
    y = df.loc[:, xlabel]
    X = df.loc[:, ylabel]
    return X, y


if __name__ == '__main__':


    #models = ['linear', 'lasso', 'multilasso', 'lassolars', 'ridge', 'ridgeCV', 'bayes', 'huber', 'poly']
    models = ['linear']
    #validation()
    for mod in models:
        clf, continuum = train_models(mod, save=True, cutoff=0.999, percent=40, plot=True, scale=False)
    #    #with open('FASMA_ML.pkl', 'rb') as f:
    #    #    clf = cPickle.load(f)
    #    #print(clf)
    #    results = test_set(clf, mod, continuum=continuum, fname='obs.lst', scale=False, mode='apogee')

    #r = ridge_all(1000, cutoff=0.999, percent=40, fname_synth='obs_synth300.lst', fname_obs='obs.lst')
    #r = ridge(0.1, cutoff=0.999, percent=40, fname='obs.lst')
    #alpha = [900, 950, 1000, 1050, 1100]
    #for a in alpha:
    #    r = ridge_all(a, cutoff=0.999, percent=40, fname='obs_synth300.lst')
