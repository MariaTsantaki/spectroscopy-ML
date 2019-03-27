from __future__ import division
from model_training import Data, Model
from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
import matplotlib.pyplot as plt
from time import time
try:
    from joblib import Parallel, delayed
    joblib_import = True
except ImportError:
    joblib_import = False

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2


class Minimizer:
    def __init__(self, flux, model: Model, p0=(5777, 4.44, 0.00, 0.0)):
        self.flux = flux
        self.model = model
        self.wavelength = model.data.wavelength
        self.p0 = p0
        self.completed = False

    def __str__(self):
        if self.completed:
            fout = f'Parameter report:\n\tTeff={int(self.parameters[0])}'
            fout += f'\n\tlogg={round(self.parameters[1], 2)}'
            fout += f'\n\t[Fe/H]={round(self.parameters[2], 2)}'
            fout += f'\n\talpha={round(self.parameters[3], 2)}'
            return fout
        return 'Run minimize_ML.minimize() to get parameters'

    def __repr__(self):
        return f'minimize_ML(flux={self.flux}, model={self.model}, p0={self.p0})'

    def print_parameters(self):
        print(self.__str__())

    def minimize(self, method='L-BFGS-B'):
        self.method = method
        self.res = minimize(self.chi2, self.p0, method=method, bounds=((3000, 7000), (1.0, 5.0), (-2.5, 0.6), (-0.5, 0.5)), tol=10e-20, options={'gtol':1e-12, 'ftol':1e-12})
        self.parameters = self.res.x
        self.completed = True
        return self.res

    def chi2(self, p, error=0.001):
        h = self.model.get_spectrum(p)
        return np.sum((self.flux - h)**2.0) / error

    def plot(self, save=False, fname=None):
        flux0 = self.model.get_spectrum(self.p0)
        flux_res = self.model.get_spectrum(self.res.x)
        plt.figure(figsize=(12, 8))
        plt.plot(self.wavelength, self.flux, label='Observation')
        plt.plot(self.wavelength, flux0, label='Initial guess', alpha=0.4)
        plt.plot(self.wavelength, flux_res, label='Final result')
        plt.plot(self.wavelength, self.flux-flux_res + 1.1, alpha=0.7, label='Difference')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        plt.grid(True)
        x1, x2 = plt.xlim()
        y1, y2 = plt.ylim()

        plt.text(x2*0.98, y1*1.08, r'$T_\mathrm{eff}$=%sK' % int(self.parameters[0]))
        plt.text(x2*0.98, y1*1.06, r'$\log g$={:.3}dex'.format(self.parameters[1]))
        plt.text(x2*0.98, y1*1.04, '[Fe/H]={:.3}dex'.format(self.parameters[2]))

        if save:
            if fname is None or not isinstance(fname, str):
                fname = 'result.pdf'
            plt.savefig(fname)
        plt.show()


if __name__ == '__main__':
    print('Loading data...')
    t = time()
    data = Data('spec_ML.hdf')
    print('Loaded data in {}s\n'.format(round(time()-t, 2)))
    print('Loading model...')
    # model = Model(data, classifier='ridgeCV', load=True, fname='FASMA_large_ML.pkl')
    model = Model(data, classifier='ridgeCV', save=True, fname='FASMA_large_ML.pkl')
    wavelength = data.wavelength
    result = data.X_test.iloc[0]
    flux = data.y_test.iloc[0]

    print('Minimizing...')
    t = time()
    minimizer = Minimizer(flux, model)
    res = minimizer.minimize()
    print('Minimized in {}s\n'.format(round(time()-t, 2)))
    minimizer.plot()

    #print('#'*30)
    #print('Teff(real) {}K'.format(int(result['teff'])))
    #print('Teff(min) {}K'.format(int(res.x[0])))
    #print('logg(real) {}dex'.format(result['logg']))
    #print('logg(min) {}dex'.format(round(res.x[1], 2)))
    #print('[Fe/H](real) {}dex'.format(result['feh']))
    #print('[Fe/H](min) {}dex'.format(round(res.x[2], 2)))
    #print('[a/Fe](real) {}dex'.format(result['alpha']))
    #print('[a/Fe](min) {}dex'.format(round(res.x[3], 2)))

    #minimizer.plot()

    if joblib_import and False:
        print('\n\nComparing running on multiple cores')
        N = int(len(data.y_test)/10)
        def f(i):
            flux = data.y_test.iloc[i]
            m = Minimizer(flux, model)
            r = m.minimize()
            return r

        t = time()
        r1 = [f(i) for i in range(N)]
        t1 = time() - t

        t = time()
        r2 = Parallel(n_jobs=3)(delayed(f)(i) for i in range(N))
        t2 = time() - t

        print('Speedup on running on 3 CPUs: X{}'.format(round(t1/t2, 2)))
