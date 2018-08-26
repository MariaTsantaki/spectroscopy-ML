from __future__ import division
from specML.model_training import Data, Model
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
    def __init__(self, flux, model, p0=(5777, 4.44, 0.00, 0.0)):
        self.flux = flux
        self.model = model
        self.p0 = p0

    def minimize(self, method=None):
        self.method = method
        self.res = minimize(self.chi2, self.p0, method=method)
        self.parameters = self.res.x
        return self.res

    def chi2(self, p, error=1):
        # print(p)
        h = self.model.get_spectrum(p)
        return np.sum((self.flux - h)**2 / error)

    def plot(self, save=False, fname=None):
        def onpick(event):
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            fig.canvas.draw()
        wavelength = self.model.data.get_wavelength()
        flux0 = self.model.get_spectrum(self.p0)
        flux_res = self.model.get_spectrum(self.res.x)
        fig = plt.figure(figsize=(12, 8))
        line1, = plt.plot(wavelength, self.flux, label='Observation')
        line2, = plt.plot(wavelength, flux0, label='Initial guess', alpha=0.4)
        line3, = plt.plot(wavelength, flux_res, label='Final result')
        line4, = plt.plot(wavelength, self.flux-flux_res + 1.1, alpha=0.7, label='Difference')
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        leg = plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        plt.grid(True)
        x1, x2 = plt.xlim()
        y1, y2 = plt.ylim()

        lines = [line1, line2, line3, line4]
        lined = dict()
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline

        plt.text(x2*0.98, y1*1.08, r'$T_\mathrm{eff}$=%sK' % int(self.parameters[0]))
        plt.text(x2*0.98, y1*1.06, r'$\log g$={:.3}dex'.format(self.parameters[1]))
        plt.text(x2*0.98, y1*1.04, '[Fe/H]={:.3}dex'.format(self.parameters[2]))

        if save:
            if fname is None:
                fname = 'result.pdf'
            plt.savefig(fname)
        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()


if __name__ == '__main__':

    # data = Data('data/spec_ml_sample.hdf', with_quadratic_terms=True)
    data = Data('data/spec_ml_sample.hdf', with_quadratic_terms=False, scale=False)
    # model = Model(data, classifier='nn', save=True, fname='FASMA_ML_nn.pkl')
    # model = Model(data, classifier='nn', load=True, fname='FASMA_ML_nn.pkl')
    model = Model(data, classifier='ridge', load=True, fname='FASMA_large_ML.pkl')
    wavelength = data.get_wavelength()
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
    print('[a/Fe](real) {}dex'.format(result['alpha']))
    print('[a/Fe](min) {}dex'.format(round(res.x[3], 2)))

    minimizer.plot()


    if joblib_import and False:
        print('\n\nComparing running on multiple cores')
        N = int(len(data.y_test)/10)
        def f(i):
            print(i)
            flux = data.y_test.iloc[i]
            result = data.X_test.iloc[i]
            m = Minimizer(flux, model)
            r, _, _ = m.minimize(method='Nelder-Mead')
            return [r.x[0], result['teff']]

        # t = time()
        # r1 = [f(i) for i in range(N)]
        # t1 = time() - t

        t1 = time()
        r2 = Parallel(n_jobs=3)(delayed(f)(i) for i in range(N))
        t2 = time() - t

        print('Speedup on running on 3 CPUs: X{}'.format(round(t1/t2, 2)))
