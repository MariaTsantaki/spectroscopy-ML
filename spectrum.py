import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
try:
    import cPickle
except ImportError:
    import _pickle as cPickle
from error import CleanError, NormalizeError


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2



# TODO: Fix this!
class _Normalize:

    def __init__(self, wavelength, flux, kind='linear', degree=2):
        self.wavelength = wavelength
        self.flux = flux
        self.kind = kind
        self.degree = degree

        if not self.kind in ['constant', 'linear', 'polynomial']:
            raise ValueError('Wrong kind of normalisation')
        if self.kind == 'linear':
            self.degree = 1  # To simplify later

    def normalize(self):
        self.getIntervals('intervals.lst')
        self.getMeanValues()

        if self.kind is 'constant':
            self.flux -= np.mean(self.meanFluxValues)
        elif (self.kind is 'linear') or (self.kind is 'polynomial'):
            p = np.polyfit(self.meanWavelengthValues, self.meanFluxValues, self.degree)
            f = np.poly1d(p)
            self.flux /= f(self.wavelength)
        return self.flux

    def getMeanValues(self):
        N = self.interval.shape[0]
        self.meanFluxValues = np.zeros(N)
        self.meanWavelengthValues = np.zeros(N)
        for i, range in enumerate(self.interval):
            self.meanFluxValues[i] = self.getMean(range)
            self.meanWavelengthValues[i] = (range[1] + range[0])/2
        idx = ~np.isnan(self.meanFluxValues)
        self.meanWavelengthValues = self.meanWavelengthValues[idx]
        self.meanFluxValues = self.meanFluxValues[idx]

    def getIntervals(self, fname):
        self.interval = np.loadtxt(fname)

    def getMean(self, range):
        idx = (range[0] <= self.wavelength) & (self.wavelength <= range[1])
        if not idx.any():
            return np.nan
        return np.mean(self.flux[idx])


class Spectrum:

    def __init__(self, wavelength, flux, star):
        if len(wavelength) != len(flux):
            raise ValueError('Length of wavelength and flux must be equal.')
        self.wavelength = wavelength
        self.flux = flux
        self.star = star  # Name of the star
        self.cleaned = False
        self.new_grid = False
        self.normalized = False
        self.model_set = False

    def clean(self):
        """Remove cosmic rays from the spectrum"""
        data = self.flux
        med = np.median(self.flux)
        self.flux[self.flux <= 0] = med
        med = np.median(self.flux)
        sigma = np.mean(np.absolute(self.flux - np.mean(self.flux, None)), None)
        self.flux[self.flux > (med + sigma*5)] = med
        self.cleaned = True

    def normalize(self, kind='linear', degree=2):
        """
        Normalize the spectrum to put on a scale as the trained model
        """
        if not self.cleaned:
            raise CleanError('Please clean the spectrum first')
        self.flux = _Normalize(self.wavelength, self.flux).normalize()
        self.normalized = True

    def interpolate(self, w, kind='linear'):
        """
        Interpolate the spectrum to the new wavelength 'w'
        """
        if (w[0] < self.wavelength[0]) or (w[-1] > self.wavelength[-1]):
            raise ValueError('New grid extend beyond the spectrum. Choose a smaller region')
        if (w[1] - w[0]) < (self.wavelength[1] - self.wavelength[0]):
            raise ValueError('Resolution of new grid is higher than the spectrum')
        if not self.normalized:
            raise NormalizeError()
        # Find the interpolation
        f = interp1d(self.wavelength, self.flux, kind=kind)
        # Apply the interpolation on new wavelength grid
        self.flux = f(w)
        self.wavelength = w
        self.new_grid = True

    def set_model(self, fname, percentage=1):
        with open(fname, 'rb') as f:
            print('Loading model...')
            self.model = cPickle.load(f)
            print('Model succesfully loaded.')
            if percentage < 1:
                print('Removing high coefficients from the model...')
                for i, pi in enumerate(self.model.coef_):
                    idx = abs(pi) <= percentage * max(abs(pi))
                    pi[~idx] = 0
                    self.model.coef_[i] = pi
                # idx = self.flux < percentage * max(self.flux)
                # self.model.coef_[:, ~idx] = 0
            print('Model ready to use. Percentage: {}%'.format(percentage*100))
            self.model_set = True

    def get_parameters(self):
        if not self.model_set:
            raise ValueError('Please set a model first with "set_model(fname)"')
        if not self.new_grid:
            raise ValueError('''Please put the spectrum on the same grid as the model. If it already is, then use: "spectrum.new_grid = True" before this method call''')
        self.parameters = self.model.predict(self.flux.reshape(1, -1))[0]
        # self.parameters = np.dot(self.model.coef_, self.flux) + self.model.intercept_
        self.parameters[0] = int(self.parameters[0])
        self.parameters[1] = round(self.parameters[1], 2)
        self.parameters[2] = round(self.parameters[2], 2)
        self.parameters[3] = round(self.parameters[3], 2)
        self.parameters[4] = round(self.parameters[4], 2)
        self.parameters[5] = round(self.parameters[5], 2)
        return self.parameters

    def print_parameters(self):
        if not 'parameters' in self.__dict__.keys():
            raise ValueError('Get the parameters first with "get_parameters()"')
        p = self.parameters
        print('\nTemperature: {} K'.format(p[0]))
        print('logg: {} dex'.format(p[1]))
        print('[Fe/H]: {} dex'.format(p[2]))
        print('vmic: {} km/s'.format(p[3]))
        print('vmac: {} km/s'.format(p[4]))
        print('vsini: {} km/s'.format(p[5]))

    def plot(self, save=False):
        plt.plot(self.wavelength, self.flux)
        plt.xlabel(r'Wavelength [$\AA]')
        plt.ylabel('Normalized flux')
        plt.title('Star: {}'.format(self.star))

        # Parameters
        x0 = self.wavelength[0]
        y = self.flux[0]
        names = ['Teff', 'logg', '[Fe/H]', 'vmic', 'vmac', 'vsini']
        for name, value in zip(names, self.parameters):
            text = '{}: {}'.format(name, value)
            plt.text(x0, y, text)
            y -= 0.03
        plt.tight_layout()
        if save:
            plt.savefig('figures/{}.pdf'.format(self.star))
        plt.show()





if __name__ == '__main__':
    import sys
    import numpy as np
    import pandas as pd
    from generate_test_spectrum import generate
    if len(sys.argv) > 1:
        if sys.argv[1] == 'real':
            df = pd.read_hdf('combined_spec.hdf')
            wavelength = np.array(list(map(float, df.keys()[:-7])))
            while True:
                dfi = df.sample(1)
                flux = dfi.values[0][:-7].astype('float')
                realParameters = dfi.values[0][-7:-1]
                star = dfi.values[0][-1]
                spectrum = Spectrum(wavelength, flux, star)
                # TODO: Either (or both) .clean or .normalize will "destroy" the
                #       spectrum so parameters cannot be obtained
                spectrum.clean()
                spectrum.normalize('constant')
                spectrum.set_model('FASMA_ML.pkl', percentage=0.01)
                spectrum.new_grid = True
                p = spectrum.get_parameters()
                for p1, p2 in zip(p, realParameters):
                    print('{}\t{}'.format(p1, p2))
                # spectrum.print_parameters()
                # print('Real parameters:', realParameters)
                spectrum.plot()
    else:
        N = 9959
        wavelength, flux = generate(15100, 15200, N)
        flux += 4

        # Initialize a spectrum object
        spectrum = Spectrum(wavelength, flux, 'My favourite star')

        # Clean the spectrum for cosmics
        spectrum.clean()

        # Normalize the spectrum
        spectrum.normalize(kind='linear')

        # Interpolate to a new wavelength scale.
        # This should be the header of the 'combined_spec.csv'
        # spectrum.interpolate(new_wavelength)

        # Set a model
        spectrum.set_model('FASMA_ML.pkl')

        # Get parameters
        spectrum.new_grid = True  # TODO: Remove this eventually...
        parameters = spectrum.get_parameters()
        spectrum.print_parameters()
        spectrum.plot()
