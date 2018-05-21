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
    xlabel = df.columns.values[:-6]
    ylabel = df.columns.values[-6:]
    y = df.loc[:, xlabel]
    X = df.loc[:, ylabel]
    return X, y


if __name__ == '__main__':
    X, y = getData()
    wavelength = np.array(list(map(float, y.columns.values)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = linear_model.LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    i = randint(0, len(X_test)-1)
    f_predict = clf.predict(X_test.iloc[i].values.reshape(1, -1))[0]
    f_test = y_test.iloc[i].values

    plt.subplot(211)
    plt.plot(wavelength, f_test, label='Synthetic')
    plt.plot(wavelength, f_predict, label='ML predict')
    plt.ylabel('Flux')
    plt.subplot(212)
    plt.plot(wavelength, f_test-f_predict, label='Synthetic')
    plt.grid(True)
    plt.xlabel('Wavelength')
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.show()
