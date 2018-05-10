from scipy.interpolate import interp1d
try:
    import cPickle
except ImportError:
    import _pickle as cPickle
from utils import normalize, mad


class Spectrum:

    def __init__(self, wavelength, flux):
        if len(wavelength) != len(flux):
            raise ValueError('Length of wavelength and flux must be equal.')
        self.wavelength = wavelength
        self.flux = flux
        self.cleaned = False
        self.new_grid = False
        self.normalized = False
        self.model = False

    def clean(self):
        """Remove cosmic rays from the spectrum"""
        med = np.median(self.flux)
        self.flux[self.flux == 0] = med
        med = np.median(self.flux)
        sigma = mad(self.flux)
        self.flux[self.flux > (med + sigma*5)] = med
        self.cleaned = True

    def normalize(self, kind='linear'):
        """
        Normalize the spectrum to put on a scale as the trained model
        """
        if not self.cleaned:
            raise ValueError('Please clean the spectrum first')
        self.flux = normalize(self.wavelength, self.flux)
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
            raise ValueError('Please normalize the spectrum first')
        # Find the interpolation
        f = interp1d(self.wavelength, self.flux, kind=kind)
        # Apply the interpolation on new wavelength grid
        self.flux = f(w)
        self.wavelength = w
        self.new_grid = True

    def set_model(self, fname):
        with open(fname, 'rb') as f:
            print('Loading model...')
            self.model = cPickle.load(f)
            print('Model succesfully loaded.')
            self.model = True

    def get_parameters(self):
        if not self.model:
            raise ValueError('Please set a model first with "set_model(fname)"')
        if not self.new_grid:
            raise ValueError('''Please put the spectrum on the same grid as the model.
            If it already is, then use: "spectrum.new_grid = True" before this method call''')
        self.parameters = self.model.predict(self.flux.reshape(1, -1))
        return self.parameters

    def print_parameters(self):
        if not 'parameters' in self.__dict__.keys():
            raise ValueError('Get the parameters first with "get_parameters()"')
        p = self.parameters[0]
        print('\nTemperature: {} K'.format(int(p[0])))
        print('logg: {} dex'.format(round(p[1], 2)))
        print('[Fe/H]: {} dex'.format(round(p[2], 2)))
        print('vmic: {} km/s'.format(round(p[3], 2)))
        print('vmac: {} km/s'.format(round(p[4], 2)))
        print('vsini: {} km/s'.format(round(p[5], 2)))


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from generate_test_spectrum import generate
    N = 9959
    wavelength, flux = generate(15100, 15200, N)
    flux += 4

    # Initialize a spectrum object
    spectrum = Spectrum(wavelength, flux)
    plt.plot(spectrum.wavelength, spectrum.flux, label='Raw spectrum')

    # Clean the spectrum for cosmics
    spectrum.clean()

    # Normalize the spectrum
    spectrum.normalize()
    plt.plot(spectrum.wavelength, spectrum.flux, label='Normalized spectrum')

    # Interpolate to a new wavelength scale.
    # This should be the header of the 'combined_spec.csv'
    # spectrum.interpolate(new_wavelength)

    # Set a model
    spectrum.set_model('FASMA_ML.pkl')

    # Get parameters
    spectrum.new_grid = True  # TODO: Remove this eventually...
    parameters = spectrum.get_parameters()
    spectrum.print_parameters()

    plt.tight_layout()
    plt.legend()
    plt.show()
