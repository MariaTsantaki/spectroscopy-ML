from model_training import Data, Model
from scipy.optimize import minimize
import numpy as np
from time import time
try:
    from joblib import Parallel, delayed
    joblib_import = True
except ImportError:
    joblib_import = False

class Minimizer:
    def __init__(self, flux, model, p0=(5777, 4.44, 0.00)):
        self.flux = flux
        self.model = model
        self.p0 = p0

    def minimize(self, max_iter=1000, tol=0.01):
        res = minimize(self.chi2, self.p0)
        return res

    def chi2(self, p, error=1):
        h = self.model.get_spectrum(p)
        return np.sum((self.flux - h)**2) / error


if __name__ == '__main__':
    data = Data('combined_spec.hdf')
    model = Model(data, classifier='linear', load=True)

    result = data.X_test.iloc[0]
    flux = data.y_test.iloc[0]

    t = time()
    minimizer = Minimizer(flux, model)
    res = minimizer.minimize()
    print('Minimized in {}s\n'.format(round(time()-t, 2)))

    print('#'*30)
    print('Teff(real) {}K'.format(int(result['teff'])))
    print('Teff(min) {}K'.format(int(res.x[0])))
    print('logg(real) {}dex'.format(result['logg']))
    print('logg(min) {}dex'.format(round(res.x[1], 2)))
    print('[Fe/H](real) {}dex'.format(result['feh']))
    print('[Fe/H](min) {}dex'.format(round(res.x[2], 2)))

    if joblib_import:
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
