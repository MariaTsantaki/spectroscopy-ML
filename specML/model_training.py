from __future__ import division
import numpy as np
from sklearn import linear_model, preprocessing, neural_network
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression, VarianceThreshold
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
            self.X['alpha**2'] = self.X['alpha'] ** 2
            self.X['teff*logg'] = self.X['teff'] * self.X['logg']
            self.X['teff*feh'] = self.X['teff'] * self.X['feh']
            self.X['logg*feh'] = self.X['logg'] * self.X['feh']
            self.X['teff*alpha'] = self.X['teff'] * self.X['alpha']
            self.X['alpha*feh'] = self.X['alpha'] * self.X['feh']
            self.X['logg*alpha'] = self.X['logg'] * self.X['alpha']
        self.labels = self.X.columns

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
        self.scaler = preprocessing.RobustScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)

    def feature_selection_percentile(self):
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        selector = SelectPercentile(f_regression, percentile=20)
        y = self.y.values
        totalscore = []
        for i, yy in enumerate(y):
            selector.fit_transform(self.X, y[:,i])
            names = [feature_names[i] for i in np.argsort(selector.scores_)[::-1]]
            totalscore.append(selector.scores_)
        #if feature selection is used then labels should change!

    def feature_selection_variance(self):
        selector = VarianceThreshold(0.5)
        selector.fit(self.X)
        features = selector.get_support(indices = True)
        feature_names = ['teff', 'logg', 'feh', 'alpha', 'teff**2', 'logg**2', 'feh**2', 'alpha**2', 'teff*logg', 'teff*feh', 'logg*feh', 'teff*alpha', 'alpha*feh', 'logg*alpha']
        for i in features:
            print(feature_names[i])
        features = [column for column in self.X[features]]


class Model:
    def __init__(self, data, classifier='linear', save=True, load=False, fname='FASMA_ML.pkl'):
        self.classifier = classifier.lower()
        self.data = data
        self.save = save
        self.load = load
        self.fname = fname
        self.X_train, self.y_train = self.data.X, self.data.y

        if self.classifier == 'linear':
            self.clf = linear_model.LinearRegression(n_jobs=-1)
        elif self.classifier == 'lasso':
            self.clf = linear_model.Lasso(alpha=0.00001)
        elif self.classifier == 'lassolars':
            self.clf = linear_model.LassoLars(alpha=1000)
        elif self.classifier == 'multilasso':
            self.clf = linear_model.MultiTaskLasso(alpha=1000)
        elif self.classifier == 'ridgeCV':
            self.clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 100])
        elif self.classifier == 'ridge':
            self.clf = linear_model.Ridge(alpha=10)
        elif self.classifier == 'bayes':
            self.clf = linear_model.BayesianRidge()
        elif self.classifier == 'huber':
            self.clf = linear_model.HuberRegressor()
        elif self.classifier == 'nn':
            self.clf = neural_network.MLPRegressor(hidden_layer_sizes=(6,), verbose=True)


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
            v += [teff**2, logg**2, feh**2, alpha**2, teff*logg, teff*feh, logg*feh, teff*alpha, alpha*feh, logg*alpha]
        v = np.array(v).reshape(1, -1)
        return v

    def get_spectrum(self, p):
        v = self._get_label_vector(p)
        if self.data.scale:
            v = self.data.scaler.transform(v)
        f = self.clf.predict(v)[0]
        f[f>1] = 1
        f[f<0] = 0
        return f


if __name__ == '__main__':
    data = Data('data/spec_ml.hdf', with_quadratic_terms=False, scale=False)
    # continuum = data.flux_removal(cutoff=0.999, percent=50)
    model = Model(data, classifier='linear', save=True, fname='FASMA_large_ML.pkl')
    # model = Model(data, classifier='nn', save=True, fname='FASMA_ML_nn.pkl')
    wavelength = data.get_wavelength()
    flux = model.get_spectrum((5320, 3.42, 0.05, 0.05))
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux)
    plt.show()
