import numpy as np
from random import choice

def generate(start, end, length, absorption_lines=35):
    """Generate a test spectrum"""

    wavelength = np.linspace(start, end, length)
    flux = np.ones_like(wavelength) + np.random.random(length) * 0.05

    for _ in range(absorption_lines):
        flux -= random_gauss(wavelength)

    return wavelength, flux


def random_gauss(x):
    a = np.random.random()
    b = choice(x)
    c = np.random.random()
    g = a*np.exp(-(x-b)**2/(2*c**2))
    return g
