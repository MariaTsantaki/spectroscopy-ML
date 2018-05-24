from model_training import Data, Model
from simple_minimization import Minimizer
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from utils import prepare_spectrum_synth, prepare_spectrum, save_and_compare_synthetic, save_and_compare_apogee

def self_check(X_test, y_test, model, plot=True):
    x_pred = []
    for i, y in enumerate(y_test.values[:]):
        minimizer = Minimizer(y, model)
        res = minimizer.minimize()
        x_pred.append([res.x[0], res.x[1], res.x[2], res.x[3]])
    xlabel = ['teff', 'logg', 'feh', 'alpha']
    params = pd.DataFrame(np.array(x_pred), columns=xlabel)
    if plot:
        for i, label in enumerate(xlabel):
            plt.figure()
            plt.scatter(X_test[label], X_test[label].values - params[label].values, s=70, alpha=0.4)
            plt.grid()
            plt.title(label)
            #plt.savefig(label + '_' + model + '.png')
            plt.show()

def test_set_synth(model, continuum=None, fname='obs_synth.lst'):

    spec = np.genfromtxt(fname, dtype='str')
    params = []
    for s in spec[:]:
        y, w = prepare_spectrum_synth(s, continuum)
        minimizer = Minimizer(y, model)
        res = minimizer.minimize()
        params.append([res.x[0], res.x[1], res.x[2], res.x[3]])
        #print('Star: %s' % s)
        #print('\nStellar atmospheric parameters:')
        #print('Teff:   {:.0f} K'.format(res.x[0]))
        #print('logg:   {:.2f} dex'.format(res.x[1]))
        #print('[M/H]:  {:.2f} dex'.format(res.x[2]))
        #print('alpha:  {:.2f} dex'.format(res.x[3]))
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
    results = save_and_compare_synthetic(d, class_name='linear')
    return results

def test_set_apogee(model, continuum=None, fname='obs.lst'):

    spec = np.genfromtxt(fname, dtype='str')
    params = []
    for s in spec[:]:
        y, w = prepare_spectrum(s, continuum)
        minimizer = Minimizer(y, model)
        res = minimizer.minimize()
        params.append([res.x[0], res.x[1], res.x[2], res.x[3]])
        #print('Star: %s' % s)
        #print('\nStellar atmospheric parameters:')
        #print('Teff:   {:.0f} K'.format(res.x[0]))
        #print('logg:   {:.2f} dex'.format(res.x[1]))
        #print('[M/H]:  {:.2f} dex'.format(res.x[2]))
        #print('alpha:  {:.2f} dex'.format(res.x[3]))
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
    results = save_and_compare_apogee(d, 'lol')
    return results


if __name__ == '__main__':
    data = Data('spec_ml.hdf')
    continuum = data.flux_removal(cutoff=0.999, percent=50)
    X_test = data.X_test
    y_test = data.y_test

    class_name = ['lasso', 'multilasso', 'lassolars', 'ridge', 'ridgeCV']
    for m in class_name:
        print(m)
        model = Model(data, classifier=m)
        self_check(X_test, y_test, model, plot=True)
        test_set_synth(model, continuum=continuum)
        #test_set_apogee(model, continuum=continuum)
