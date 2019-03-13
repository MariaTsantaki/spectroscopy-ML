from __future__ import division
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
try:
    import cPickle
except ImportError:
    import _pickle as cPickle


class Data:
    def __init__(self, fname, split=False, scale=True, feature=True):
        self.fname   = fname
        self.split   = split
        self.scale   = scale
        self.feature = feature

        if self.fname.endswith('.hdf'):
            reader = pd.read_hdf
        elif self.fname.endswith('.csv'):
            reader = pd.read_csv
        self.df = reader(fname, index_col=0)
        self.df = self.df.apply(pd.to_numeric)
        self.df = self.df.reset_index(drop=True)

        self._prepare_data()
        if self.split:
            self.split_data()
        if self.scale:
            self.scale_data()
        if self.feature:
            self.feature_selection_variance()
        #X.data, y.data are the trained data, X_test and y_test are not scaled!
        print('The number of spectra included is %s ' % (len(self.X)))

    def get_wavelength(self):
        wavelength = np.array(list(map(float, self.y.columns.values)))
        return wavelength

    def _prepare_data(self):
        self.xlabel = ['teff', 'logg', 'feh', 'alpha']
        self.ylabel = self.df.columns.values[:-7]
        self.X = self.df.loc[:, self.xlabel]
        self.y = self.df.loc[:, self.ylabel]
        self.Xml = self.X.values
        self.yml = self.y.values

    def flux_removal(self, cutoff=0.998, percent=40):
        #print('The percentage of flux points dropped is %s with a %s cutoff.' % (percent, cutoff))
        continuum = []
        for wavelength in self.ylabel[:]:
            flux = self.y[wavelength]
            flux_cont = flux.loc[flux > cutoff]
            if (len(flux_cont)/len(flux))*100 > percent:
                continuum.append(wavelength)
        columns = np.array(continuum)
        self.y.drop(columns, inplace=True, axis=1)
        self.yml = self.y.values
        #self.y_test.drop(columns, inplace=True, axis=1, errors='ignore')
        print('The number of flux points is %s from the original %s.' % (len(self.ylabel)-len(continuum), len(self.ylabel)))
        return columns

    def split_data(self, test_size=0.0001):
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

    def scale_data(self):
        '''Available scalers : MinMaxScaler, MaxAbsScaler, StandardScaler,
        RobustScaler, Normalizer, QuantileTransformer, PowerTransformer'''

        self.scaler = preprocessing.StandardScaler().fit(self.Xml)
        self.Xml = self.scaler.transform(self.Xml)

    def feature_selection_variance(self):
        selector = VarianceThreshold(0.16)
        selector.fit(self.Xml)
        features = selector.get_support(indices=True)
        f = [self.xlabel[x] for x in features]
        self.Xml = self.Xml[:, features]
        self.xlabel = f
        self.flabel = features
        lol = list(f)
        print('The features used are %s: ' % (len(lol)))
        print(', '.join(lol))

class Model:
    def __init__(self, data, classifier='linear', save=True, load=False, alpha=0.01, fname='FASMA_ML.pkl'):
        self.classifier = classifier
        self.data  = data
        self.save  = save
        self.load  = load
        self.alpha = alpha
        self.fname = fname
        self.X_train, self.y_train = data.Xml, data.yml

        if self.classifier == 'linear':
            self.clf = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression(n_jobs=-1))])
        elif self.classifier == 'lasso':
            self.clf = Pipeline([('poly', PolynomialFeatures(degree=2)), ('lasso', linear_model.Lasso(alpha=0.0001, max_iter=5000))])
        elif self.classifier == 'lassolars':
            self.clf = Pipeline([('poly', PolynomialFeatures(degree=2)), ('lassolars', linear_model.LassoLars(alpha=1e-06))])
        elif self.classifier == 'multilasso':
            self.clf = Pipeline([('poly', PolynomialFeatures(degree=2)), ('multilasso', linear_model.MultiTaskLasso(alpha=1e-06))])
        elif self.classifier == 'ridge':
            self.clf = Pipeline([('poly', PolynomialFeatures(degree=2)), ('ridge', linear_model.Ridge(alpha=alpha))])
        print(self.clf)
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
        if self.data.feature:
            v = np.array(v)
            v = v[self.data.flabel]
        v = np.array(v).reshape(1, -1)
        return v

    def get_spectrum(self, p):
        v = self._get_label_vector(p)
        if self.data.scale:
            v = self.data.scaler.transform(v)
        f = self.clf.predict(v)[0]
        return f

    def predict_output(self, p):
        f = self.clf.predict(p)
        return f


if __name__ == '__main__':
    from astropy.io import fits

    data = Data('spec_ML_marcs.hdf', scale=True, feature=False)
    wavelength = data.get_wavelength()
    data.flux_removal(cutoff=0.999, percent=50)
    model = Model(data, classifier='linear', load=False)
    teff, logg, feh, alpha = 5777, 4.44, 0, 0.0
    flux = model.get_spectrum((teff, logg, feh, alpha))
    plt.plot(wavelength, flux, label='marcs')
    plt.legend()
    plt.show()
