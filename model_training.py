from __future__ import division
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import pandas as pd
try:
    import cPickle
except ImportError:
    import _pickle as cPickle


class Data:
    def __init__(self, fname, with_quadratic_terms=True, split=True, scale=True):
        self.fname = fname
        self.with_quadratic_terms = with_quadratic_terms
        self.split = split
        self.scale = scale

        if self.fname.endswith('.hdf'):
            reader = pd.read_hdf
        elif self.fname.endswith('.csv'):
            reader = pd.read_csv
        self.df = reader(fname, index_col=0)
        self.df.set_index('spectrum', inplace=True)

        self._prepare_data()
        if self.split:
            self.split_data()
        if self.scale:
            self.scale_data()

        #X.data,y.data are the trained data, X_test and y_test are not scaled!

    def get_wavelength(self):
        wavelength = np.array(list(map(float, self.y.columns.values)))
        return wavelength

    def _prepare_data(self):
        self.xlabel = ['teff', 'logg', 'feh', 'alpha']
        self.ylabel = self.df.columns.values[:-7]
        self.X = self.df.loc[:, self.xlabel]
        self.y = self.df.loc[:, self.ylabel]

        if self.with_quadratic_terms:
            self.X['teff**2'] = self.X['teff'] ** 2
            self.X['logg**2'] = self.X['logg'] ** 2
            self.X['feh**2'] = self.X['feh'] ** 2
            self.X['teff*logg'] = self.X['teff'] * self.X['logg']
            self.X['teff*feh'] = self.X['teff'] * self.X['feh']
            self.X['logg*feh'] = self.X['logg'] * self.X['feh']

    def flux_removal(self, cutoff=0.995, percent=40):
        print('The percentage of flux points dropped is %s with a %s cutoff.' % (percent, cutoff))
        continuum = []
        for wavelength in self.ylabel[:]:
            flux = self.y[wavelength]
            flux_cont = flux.loc[flux > cutoff]
            if (len(flux_cont)/len(flux))*100 > percent:
                continuum.append(wavelength)
        columns = np.array(continuum)
        self.y.drop(columns, inplace=True, axis=1)
        self.y_test.drop(columns, inplace=True, axis=1)
        print('The number of flux points is %s from the original %s.' % (len(self.ylabel)-len(continuum), len(self.ylabel)))
        return continuum

    def split_data(self, test_size=0.10):
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

    def scale_data(self):
        self.scaler = preprocessing.StandardScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)


class Model:
    def __init__(self, data, classifier='linear', save=True, load=False, fname='FASMA_ML.pkl'):
        self.classifier = classifier
        self.data = data
        self.save = save
        self.load = load
        self.fname = fname
        self.X_train, self.y_train = data.X, data.y

        if self.classifier == 'linear':
            self.clf = linear_model.LinearRegression(n_jobs=-1)
        elif self.classifier == 'lasso':
            self.clf = linear_model.Lasso(alpha=0.001)
        elif self.classifier == 'lassolars':
            self.clf = linear_model.LassoLars(alpha=0.001)
        elif self.classifier == 'multilasso':
            self.clf = linear_model.MultiTaskLasso(alpha=0.1)
        elif self.classifier == 'ridgeCV':
            self.clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        elif self.classifier == 'ridge':
            self.clf = linear_model.Ridge(alpha=0.01)
        elif self.classifier == 'bayes':
            self.clf = linear_model.BayesianRidge()
        elif self.classifier == 'huber':
            self.clf = linear_model.HuberRegressor()

        # Train the classifier
        if not self.load:
            t = time()
            self.train_classifier()
            print('Trained classifier in {}s'.format(round(time()-t, 2)))
        else:
            with open(self.fname, 'rb') as f:
                self.clf = cPickle.load(f)

    def train_classifier(self):
        self.clf.fit(self.X_train, self.y_train)
        if self.save:
            with open(self.fname, 'wb') as f:
                cPickle.dump(self.clf, f)

    def _get_label_vector(self, p):
        teff, logg, feh, alpha = p
        v = [teff, logg, feh, alpha]
        if self.data.with_quadratic_terms:
            v += [teff**2, logg**2, feh**2, teff*logg, teff*feh, logg*feh]
        v = np.array(v).reshape(1, -1)
        return v

    def get_spectrum(self, p):
        v = self._get_label_vector(p)
        if self.data.scale:
            v = self.data.scaler.transform(v)
        f = self.clf.predict(v)[0]
        return f


if __name__ == '__main__':
    data = Data('spec_ML.csv')
    continuum = data.flux_removal(cutoff=0.999, percent=50)
    model = Model(data, classifier='linear')
    wavelength = data.get_wavelength()
    flux = model.get_spectrum((6320, 3.42, -0.45, 0.05))
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux)
    plt.show()
