from __future__ import division
import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
try:
    import cPickle
except ImportError:
    import _pickle as cPickle
import argparse

from utils import create_combined, prepare_linelist

#wavelengths = pd.read_csv('linelist.lst', delimiter=r'\s+', usecols=('WL',))
#wavelengths = list(map(lambda x: round(x[0], 2), wavelengths.values))
#wavelengths += ['teff', 'logg', 'feh', 'vt']


def _parser():
    parser = argparse.ArgumentParser(description='Spectroscopic parameters with ML')
    parser.add_argument('-s', '--spectrum',
                        help='Spectrum to analyze')
    parser.add_argument('-l', '--linelist',
                        help='Line list to analyze')
    parser.add_argument('-t', '--train',
                        help='Retrain the classifier',
                        default=False, action='store_true')
    parser.add_argument('-c', '--classifier',
                        help='Which classifier to use',
                        choices=('linear', 'ridge', 'lasso'),
                        default='linear')
    parser.add_argument('--save',
                        help='Save the re-trained model',
                        default=False, action='store_true')
    parser.add_argument('--plot',
                        help='Plot the results from training a new model',
                        default=False, action='store_true')
    args = parser.parse_args()
    if not args.spectrum and not args.linelist and not args.train:
        print(parser.print_help())
        raise SystemExit
    return args


def train(clf, save=True, plot=True):
    if not os.path.isfile('combined_spec.hdf'):
        create_combined()
    df = pd.read_hdf('combined_spec.hdf')
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[:-6]
    ylabel = df.columns.values[-6:]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    t = time()
    clf.fit(X_train, y_train)
    print('Trained on {} spectra in {}s\n'.format(len(df), round(time()-t, 2)))

    N = len(y_test)
    t = time()
    y_pred = clf.predict(X_test)
    t = time()-t
    speedup = 60*N/t
    print('Calculated parameters for {} stars in {:.2f}ms'.format(N, t*1e3))
    print('Speedup: {} million times'.format(int(speedup/1e6)))

    for i, label in enumerate(ylabel):
        score = mean_absolute_error(y_test[label], y_pred[:, i])
        print('Mean absolute error for {}: {:.2f}'.format(label, score))
        if plot:
            plt.figure()
            plt.plot(y_test[label], y_test[label].values - y_pred[:, i], 'o')
            plt.grid()
            plt.title(label)

    if save:
        with open('FASMA_ML.pkl', 'wb') as f:
            cPickle.dump(clf, f)
    return clf


if __name__ == '__main__':
    args = _parser()

    if args.train:
        if args.classifier == 'linear':
            clf = linear_model.LinearRegression()
        elif args.classifier == 'ridge':
            clf = linear_model.RidgeCV(alphas=[100.0, 0.01, 0.1, 1.0, 10.0])
        elif args.classifier == 'lasso':
            clf = linear_model.LassoLars(alpha=0.001)
        clf = train(clf, save=args.save, plot=args.plot)
    else:
        with open('FASMA_ML.pkl', 'rb') as f:
            clf = cPickle.load(f)

    if args.spectrum:
        raise SystemExit('Please run ARES yourself. This is difficult enough')
    elif args.linelist:
        df = pd.read_csv('combined.csv')
        df.dropna(axis=1, inplace=True)
        wavelengths = np.array(map(lambda x: round(float(x), 2), df.columns[1:-4]))
        x = prepare_linelist(args.linelist, wavelengths=wavelengths)
        p = clf.predict(x)[0]
        print('\nStellar atmospheric parameters:')
        print('Teff:   {:.0f}K'.format(p[0]))
        print('logg:   {:.2f}dex'.format(p[1]))
        print('[Fe/H]: {:.2f}dex'.format(p[2]))
        print('vt:     {:.2f}km/s'.format(p[3]))

    if args.train and args.plot:
        plt.show()
