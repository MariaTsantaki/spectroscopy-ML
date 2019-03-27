from __future__ import division
from typing import Tuple, Any
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression, VarianceThreshold
from sklearn.feature_selection import chi2, SelectKBest, RFE
try:
    import cPickle
except ImportError:
    import _pickle as cPickle


class Data:
    def __init__(self, fname, with_quadratic_terms=True, split=True, scale=True, feature=True):
        self.fname = fname
        self.with_quadratic_terms = with_quadratic_terms
        self.split = split
        self.scale = scale
        self.feature = feature

        if self.fname.endswith('.hdf'):
            reader = pd.read_hdf
        elif self.fname.endswith('.csv'):
            reader = pd.read_csv
        self.df = reader(fname, index_col=0)
        self.df = self.df.apply(pd.to_numeric)
        self.df = self.df.reset_index(drop=True)

        self._prepare_data()
        if self.feature:
            self.feature_selection_rfe()
        if self.split:
            self.split_data()
        if self.scale:
            self.scale_data()

    @property
    def wavelength(self):
        return np.array(list(map(float, self.y.columns.values)))

    def _prepare_data(self):
        self.xlabel = ['teff', 'logg', 'feh', 'alpha']
        self.ylabel = self.df.columns.values[:-7]
        self.X = self.df.loc[:, self.xlabel]
        self.y = self.df.loc[:, self.ylabel]

        if self.with_quadratic_terms:
            self.xlabel = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
            self.X['teff**2']    = self.X['teff'] ** 2.0
            self.X['logg**2']    = self.X['logg'] ** 2.0
            self.X['feh**2']     = self.X['feh'] ** 2.0
            self.X['alpha**2']   = self.X['alpha'] ** 2.0
            self.X['teff*logg']  = self.X['teff'] * self.X['logg']
            self.X['teff*feh']   = self.X['teff'] * self.X['feh']
            self.X['logg*feh']   = self.X['logg'] * self.X['feh']
            self.X['teff*alpha'] = self.X['teff'] * self.X['alpha']
            self.X['alpha*feh']  = self.X['alpha'] * self.X['feh']
            self.X['logg*alpha'] = self.X['logg'] * self.X['alpha']

    def flux_removal(self, cutoff=0.998, percent=40):
        continuum = []
        for wavelength in self.ylabel[:]:
            flux = self.y[wavelength]
            flux_cont = flux.loc[flux > cutoff]
            if (len(flux_cont)/len(flux))*100 > percent:
                continuum.append(wavelength)
        columns = np.array(continuum)
        self.y.drop(columns, inplace=True, axis=1)
        self.y_test.drop(columns, inplace=True, axis=1, errors='ignore')
        print('The number of flux points is %s from the original %s.' % (len(self.ylabel)-len(continuum), len(self.ylabel)))
        return continuum

    def split_data(self, test_size=0.0001):
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

    def scale_data(self):
        '''Available scalers : MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler,
        RobustScaler, Normalizer, QuantileTransformer, PowerTransformer'''

        self.scaler = preprocessing.RobustScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)

    def feature_selection_best(self, k=9):
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        y = self.y.values
        k = k + 1
        selector = SelectKBest(score_func=f_regression, k=k)
        totalscore = []
        for i in range(2400):
            selector.fit(self.X, y[:, i])
            x = selector.transform(self.X, y[:, i])
            print(x.shape)
            names = [feature_names[i] for i in np.argsort(selector.scores_)[:-k:-1]]
            totalscore.append(names)
        flat_names = [item for sublist in totalscore for item in sublist]
        a = pd.Series(flat_names).value_counts()
        columns = a.index[:k].values
        self.ix = np.isin(self.xlabel, columns, invert=True)
        index = np.where(self.ix)
        self.xlabel = np.delete(self.xlabel, index)
        self.X = self.X.loc[:, self.xlabel]

    def feature_selection_percentile(self, percentile=30):
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        y = self.y.values
        totalscore = []
        selector = SelectPercentile(f_regression, percentile=percentile)
        for i in range(2400):
            x = selector.fit_transform(self.X, y[:, i])
            k = x.shape[1] + 1
            names = [feature_names[i] for i in np.argsort(selector.scores_)[:-k:-1]]
            totalscore.append(names)
        flat_names = [item for sublist in totalscore for item in sublist]
        columns = np.unique(flat_names)
        self.ix = np.isin(self.xlabel, columns, invert=True)
        index = np.where(self.ix)
        self.xlabel = np.delete(self.xlabel, index)
        self.X = self.X.loc[:, self.xlabel]

    def feature_selection_variance(self):
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        selector = VarianceThreshold(0.35)
        selector.fit(self.X)
        features = selector.get_support(indices=True)
        f = [feature_names[x] for x in features]
        self.ix = np.isin(self.xlabel, f, invert=True)
        self.xlabel = np.array(f)
        self.X = self.X.loc[:, self.xlabel]

    def feature_selection_rfe(self, n_features_to_select=9):
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        feature_names = np.array(feature_names)
        y = self.y.values
        totalscore = []
        model = linear_model.LinearRegression()
        selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
        for i in range(2400):
            selector.fit_transform(self.X, y[:, i])
            ind = selector.support_
            names = feature_names[ind]
            totalscore.append(names)
        flat_names = [item for sublist in totalscore for item in sublist]
        columns = np.unique(flat_names)
        self.ix = np.isin(self.xlabel, columns, invert=True)
        index = np.where(self.ix)
        self.xlabel = np.delete(self.xlabel, index)
        self.X = self.X.loc[:, self.xlabel]

class ShadowData:
    def __init__(self, *args, **kwargs):
        self.with_quadratic_terms = kwargs.get('with_quadratic_terms')
        self.feature = kwargs.get('feature')
        self.ix = kwargs.get('ix')
        self.scale = kwargs.get('scale')
        self.scaler = kwargs.get('scaler')
        self.wavelength = kwargs.get('wavelength')

class Model:
    def __init__(self, data=None, classifier='linear', save=True, load=False, train=True, fname='FASMA_ML.pkl'):
        self.classifier = classifier
        self.data = data
        self.save = save
        self.fname = fname
        if data is not None:
            self.X_train, self.y_train = data.X, data.y
            self.xlabel = data.xlabel

        if self.classifier == 'linear':
            self.clf = linear_model.LinearRegression(n_jobs=-1, normalize=True)
        elif self.classifier == 'lasso':
            self.clf = linear_model.Lasso(alpha=0.001, tol=0.001, max_iter=1000)
        elif self.classifier == 'lassolars':
            self.clf = linear_model.LassoLars(alpha=0.1)
        elif self.classifier == 'multilasso':
            self.clf = linear_model.MultiTaskLasso(alpha=0.1)
        elif self.classifier == 'ridgeCV':
            self.clf = linear_model.RidgeCV(alphas=[0.1, 1., 10., 100.])
        elif self.classifier == 'ridge':
            self.clf = linear_model.Ridge(alpha=0.0001)
        elif self.classifier == 'bayes':
            self.clf = linear_model.BayesianRidge()
        elif self.classifier == 'huber':
            self.clf = linear_model.HuberRegressor()

        # Train the classifier
        if not load or train:
            t = time()
            self.train_classifier()
            print('Trained classifier in {}s'.format(round(time()-t, 2)))
        else:
            self.clf, self.data = Model.load_model(self.fname)

    @staticmethod
    def load_model(fname='FASMA_ML.pkl'):
        with open(fname, 'rb') as f:
            return cPickle.load(f)

    @classmethod
    def open_model(cls, fname='FASMA_large_ML.pkl'):
        return cls(save=False, load=True, train=False, fname=fname)

    def train_classifier(self):
        self.clf.fit(self.X_train, self.y_train)
        if self.save:
            kw = {'with_quadratic_terms': self.data.with_quadratic_terms,
                  'feature': self.data.feature,
                  'ix': self.data.ix,
                  'scale': self.data.scale,
                  'scaler': self.data.scaler,
                  'wavelength': self.data.wavelength}
            data = ShadowData(**kw)
            with open(self.fname, 'wb') as f:
                cPickle.dump((self.clf, data), f)

    def _get_label_vector(self, p):
        teff, logg, feh, alpha = p
        v = [teff, logg, feh, alpha]
        if self.data.with_quadratic_terms:
            v += [teff**2.0, logg**2.0, feh**2.0, alpha**2.0, teff*logg, teff*feh, logg*feh, teff*alpha, alpha*feh, logg*alpha]
        if self.data.feature:
            l = np.invert(self.data.ix)
            v = np.array(v)
            v = v[l]
        v = np.array(v).reshape(1, -1)
        return v

    def get_spectrum(self, p):
        v = self._get_label_vector(p)
        if self.data.scale:
            v = self.data.scaler.transform(v)
        f = self.clf.predict(v)[0]
        return f


if __name__ == '__main__':
    ### Train a model from scratch
    data = Data('spec_ML.hdf', with_quadratic_terms=True, split=True, scale=True)
    data.flux_removal(cutoff=0.999, percent=10)
    model = Model(data, classifier='linear', load=False)
    wavelength = data.wavelength
    teff, logg, feh, alpha = 6800, 4.2, -0.6, 0.22
    flux = model.get_spectrum((teff, logg, feh, alpha))

    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux, 'o', label='scaled')
    plt.legend()
    plt.show()

    ### Load a model (simple way)
    clf, data = Model.load_model()
    
    ### Load a model into Model object
    model = Model.open_model('FASMA_large_ML.pkl')
