import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model as lm

from spectrum import Spectrum

def read_data(fname='combined_spec.hdf'):
    df = pd.read_hdf('combined_spec.hdf')
    df.set_index('spectrum', inplace=True)
    xlabel = df.columns.values[:-6]
    ylabel = df.columns.values[-6:]
    X = df.loc[:, xlabel]
    y = df.loc[:, ylabel]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test


def getSpectrum(wavelength, flux):
    spectrum = Spectrum(wavelength, flux, 'star')
    spectrum.clean()
    # spectrum.normalize('constant')
    return spectrum


print('Reading data...')
X_train, X_test, y_train, y_test = read_data()
wavelength = np.array(list(map(float, X_test.columns.values)))
print('Scaling and create model...')
scaler = preprocessing.StandardScaler().fit(X_train)
clf = lm.Lasso()
clf = lm.LinearRegression()

clfs = {'LinearRegression': lm.LinearRegression(n_jobs=-1),
        # 'Lasso': lm.Lasso(alpha=10),
        'RidgeCV': lm.RidgeCV(alphas=10.**np.arange(-3, 5), store_cv_values=True)}

for clf in clfs:
    print('Constructing pipeline...')
    pipeline = Pipeline([('scaling', scaler), ('model', clfs[clf])])
    print('Fit the data...')
    pipeline.fit(X_train, y_train)
    # pipeline.set_params(lasso__alpha=0.01).fit(X_train, y_train)

    # print('Make prediction...')
    # prediction = pipeline.predict(X_test)
    # NOTE: Can also be (for a single one)
    # prediction.predict(X_test.values[42].reshape(1, -1))
    print('Calculate the score of: {}...'.format(clf))
    score = pipeline.score(X_test, y_test)
    print('Score: {}%'.format(round(score*100, 2)))
    print('## Testing spectrum class ##')
    idx = X_test.sample(1).index
    flux = X_test.loc[idx]
    result = y_test.loc[idx]
    scoreBefore = pipeline.predict(flux) - result
    spectrum = getSpectrum(wavelength, flux.values[0])
    scoreAfter = pipeline.predict(spectrum.flux.reshape(1, -1)) - result
    p = (scoreAfter-scoreBefore)/scoreBefore*100
    print('Score spectrum class: {}\n'.format(p.values))
