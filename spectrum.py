try:
    import cPickle
except ImportError:
    import _pickle as cPickle


class Spectrum:

    def __init__(self, wavelength, flux):
        if len(wavelength) != len(flux):
            raise ValueError('Length of wavelength and flux must be equal.')
        self.wavelength = wavelength
        self.flux = flux
        self.new_grid = False
        self.normalize = False

    def interpolate(self, w):
        """
        Interpolate the spectrum to the new wavelength 'w'
        """
        # Find the interpolation
        # Apply it, and set:
        #   self.flux = new_flux
        #   self.wavelength = w
        #   self.new_grid = True
        raise NotImplemented('Someone do this')

    def normalize(self):
        """
        Normalize the spectrum to put on a scale as the trained model
        """
        # Good luck
        #   self.normalize = True
        raise NotImplemented('Someone do this')

    def set_model(self, fname):
        with open(fname, 'rb') as f:
            print('Loading model...')
            self.model = cPickle.load(f)
            print('Model succesfully loaded.')

    def get_parameters(self):
        if 'model' not in self.__dict__.keys():
            raise ValueError('Please set a model first with "set_model(fname)"')
        # if not self.normalize:
        #     raise ValueError('Please normalize the spectrum first')
        # if not self.new_grid:
        #     raise ValueError('Please put the spectrum on the same grid as the model')
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
    N = 9959
    wavelength = np.linspace(4500, 5500, N)
    flux = np.random.rand(N)

    # Initialize a spectrum object
    spectrum = Spectrum(wavelength, flux)

    # Interpolate to a new wavelength scale.
    # This should be the header of the 'combined_spec.csv'
    # spectrum.interpolate(new_wavelength)

    # Normalize the spectrum
    # spectrum.normalize()

    # Set a model
    spectrum.set_model('FASMA_ML.pkl')

    # Get parameters
    parameters = spectrum.get_parameters()
    spectrum.print_parameters()
