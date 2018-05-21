import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
from random import randint



def getData():
    df = pd.read_hdf('combined_spec.hdf')
    df.set_index('spectrum', inplace=True)
    ylabel = df.columns.values[:-6]
    xlabel = df.columns.values[-6:]
    xlabel = ['teff', 'logg', 'feh']
    X = df.loc[:, xlabel]
    X['teff**2'] = X['teff'] ** 2
    X['logg**2'] = X['logg'] ** 2
    X['feh**2'] = X['feh'] ** 2
    X['teff*logg'] = X['teff'] * X['logg']
    X['teff*feh'] = X['teff'] * X['feh']
    X['logg*feh'] = X['logg'] * X['feh']
    y = df.loc[:, ylabel]
    return X, y


def generateLabelVector(p):
    teff, logg, feh = p
    v = np.array([teff, logg, feh,
                  teff**2, logg**2, feh**2,
                  teff*logg, teff*feh, logg*feh])
    return v.reshape(1, -1)


def getSpectrum(clf, labelVector):
    f = clf.predict(labelVector)[0]
    return f


if __name__ == '__main__':
    X, y = getData()
    wavelength = np.array(list(map(float, y.columns.values)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = linear_model.LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    i = randint(0, len(X_test)-1)
    f_predict = clf.predict(X_test.iloc[i].values.reshape(1, -1))[0]
    f_test = y_test.iloc[i].values

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(211)
    ax1.plot(wavelength, f_test, label='Synthetic')
    ax1.plot(wavelength, f_predict, label='ML predict')
    ax1.set_xlim(15700, 15900)
    plt.legend(frameon=False)
    plt.ylabel('Flux')
    plt.subplot(212, sharex=ax1)
    plt.plot(wavelength, f_test-f_predict)
    plt.grid(True)
    plt.xlabel('Wavelength')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, getSpectrum(clf, generateLabelVector((5777, 4.44, 0.0))), label='Sun: [Fe/H]=0.00')
    plt.plot(wavelength, getSpectrum(clf, generateLabelVector((5777, 4.44, -1.0))), label='Sun: [Fe/H]=-1.00')
    plt.plot(wavelength, getSpectrum(clf, generateLabelVector((5777, 4.44, -2.0))), label='Sun: [Fe/H]=-2.00')
    plt.xlim(15700, 15800)
    plt.legend(frameon=False)


    plt.show()
