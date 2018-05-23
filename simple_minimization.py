from model_training import Data, Model
from scipy.optimize import minimize
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
    def __init__(self, flux, model, p0=(5777, 4.44, 0.00)):
        self.flux = flux
        self.model = model
        self.p0 = p0

    def minimize(self, method=None):
        self.method = method
        self.res = minimize(self.chi2, self.p0, method=method)
        return self.res

    def chi2(self, p, error=1):
        h = self.model.get_spectrum(p)
        return np.sum((self.flux - h)**2) / error

    def plot(self, save=False, fname=None):
        flux0 = self.model.get_spectrum(self.p0)
        flux_res = self.model.get_spectrum(self.res.x)
        plt.figure(figsize=(12, 8))
        plt.plot(wavelength, self.flux, label='Observation')
        plt.plot(wavelength, flux0, label='Initial guess', alpha=0.4)
        plt.plot(wavelength, flux_res, label='Final result')
        plt.plot(wavelength, flux-flux_res + 1.1, alpha=0.7, label='Difference')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        plt.grid(True)
        if save:
            if fname is None:
                fname = 'result.pdf'
            plt.savefig(fname)
        plt.show()



if __name__ == '__main__':
    data = Data('combined_spec.hdf')
    model = Model(data, classifier='linear', load=True)
    wavelength = data.get_wavelength()
    result = data.X_test.iloc[0]
    flux = data.y_test.iloc[0]

    t = time()
    minimizer = Minimizer(flux, model)
    res = minimizer.minimize(method='Nelder-Mead')
    print('Minimized in {}s\n'.format(round(time()-t, 2)))

    print('#'*30)
    print('Teff(real) {}K'.format(int(result['teff'])))
    print('Teff(min) {}K'.format(int(res.x[0])))
    print('logg(real) {}dex'.format(result['logg']))
    print('logg(min) {}dex'.format(round(res.x[1], 2)))
    print('[Fe/H](real) {}dex'.format(result['feh']))
    print('[Fe/H](min) {}dex'.format(round(res.x[2], 2)))

    minimizer.plot()


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
