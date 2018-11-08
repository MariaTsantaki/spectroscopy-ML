from __future__ import division
import numpy as np
from sklearn import linear_model, preprocessing
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
        self.df = self.df.apply(pd.to_numeric)
        #self.df.set_index('spectrum', inplace=True)
        #self.df = self.df[self.df['teff'] < 6800]
        #self.df = self.df[self.df['feh'] > -2.0]
        self.df = self.df.reset_index(drop=True)

        self._prepare_data()
        if self.split:
            self.split_data()
        if self.scale:
            self.scale_data()
        #self.feature_selection()
        #X.data, y.data are the trained data, X_test and y_test are not scaled!

    def get_wavelength(self):
        wavelength = np.array(list(map(float, self.y.columns.values)))
        return wavelength

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
            #self.X['1'] = pd.DataFrame(data=np.ones(len(self.X)))

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
        self.y_test.drop(columns, inplace=True, axis=1, errors='ignore')
        print('The number of flux points is %s from the original %s.' % (len(self.ylabel)-len(continuum), len(self.ylabel)))
        return continuum

    def split_data(self, test_size=0.01):
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
        #if feature selection is used then labels should change!


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
            self.clf = linear_model.Lasso(alpha=0.0001,  tol=0.001)
        elif self.classifier == 'lassolars':
            self.clf = linear_model.LassoLars(alpha=0.1)
        elif self.classifier == 'multilasso':
            self.clf = linear_model.MultiTaskLasso(alpha=1000)
        elif self.classifier == 'ridgeCV':
            self.clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 100])
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

        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 0.9, len(data.xlabel))])
        for i, x in enumerate(data.xlabel):
            plt.plot(self.clf.coef_.T[i], label=data.xlabel[i])
        plt.legend()
        plt.show()
        plt.plot(self.clf.intercept_)
        plt.show()

    def train_classifier(self):
        self.clf.fit(self.X_train, self.y_train)
        if self.save:
            with open(self.fname, 'wb') as f:
                cPickle.dump(self.clf, f)

    def _get_label_vector(self, p):
        teff, logg, feh, alpha = p
        v = [teff, logg, feh, alpha]
        if self.data.with_quadratic_terms:
            v += [teff**2.0, logg**2.0, feh**2.0, alpha**2.0, teff*logg, teff*feh, logg*feh, teff*alpha, alpha*feh, logg*alpha]
        v = np.array(v).reshape(1, -1)
        return v

    def get_spectrum(self, p):
        v = self._get_label_vector(p)
        if self.data.scale:
            v = self.data.scaler.transform(v)
        f = self.clf.predict(v)[0]
        return f


if __name__ == '__main__':

    data = Data('data/spec_ML.hdf', with_quadratic_terms=True, split=True, scale=True)
    data.flux_removal(cutoff=0.999, percent=30)
    model = Model(data, classifier='linear', load=False)
    wavelength = data.get_wavelength()
    teff, logg, feh, alpha = 6800, 4.2, -0.6, 0.22
    flux = model.get_spectrum((teff, logg, feh, alpha))

    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, flux)
    plt.show()
