import os
import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
from observations import read_obs_intervals


def create_combined():
    # read synthetic fluxes
    spectra = glob('synthetic_spec/results/*.spec')
    spectra = list(map(lambda x: x.split('/')[-1], spectra))

    fname = 'combined_specT.csv'
    if os.path.isfile(fname):
        os.remove(fname)

    with open(fname, 'a') as f:
        for i, spectrum in enumerate(spectra):
            d = fits.getdata('synthetic_spec/results/{}'.format(spectrum))
            if i == 0:
                wavelengths = list(map(str, d['wavelength']))
                header = ','.join(wavelengths) + ',teff,logg,feh,vmic,vmac,vsini,spectrum\n'
                f.write(header)
            else:
                flux = list(map(str, d['flux']))
                teff  = spectrum.split('_')[0]
                logg  = spectrum.split('_')[1]
                feh   = spectrum.split('_')[2]
                vmic  = spectrum.split('_')[3]
                vmac  = spectrum.split('_')[4]
                vsini = spectrum.split('_')[5]
                data = ','.join(flux + [teff, logg, feh, vmic, vmac, vsini, spectrum])
                f.write(data + '\n')


def find_star(star):
    linelists = glob('linelist/*.moog')
    linelists = list(map(lambda x: x[9:], linelists))

    affixes = ('', '_rv', '_rv2')
    for affix in affixes:
        fname = '{}{}.moog'.format(star, affix)
        if fname in linelists:
            return 'linelist/{}'.format(fname)
    raise IOError('{} not found'.format(star))


def read_star(fname):
    columns = ('wavelength', 'element', 'EP', 'loggf', 'EW')
    df = pd.read_csv(fname, delimiter=r'\s+',
                     names=columns,
                     skiprows=1,
                     usecols=['wavelength', 'EW'])
    return df


def add_parameters(df_all, df, star):
    for parameter in ('teff', 'logg', 'feh', 'vt'):
        df_all.loc[star, parameter] = df.loc[star, parameter]
    return df_all


def merge_linelist(df_all, df, star):
    for wavelength in df['wavelength']:
        df_all.loc[star, wavelength] = df[df['wavelength']==wavelength]['EW'].values[0]
    return df_all


def prepare_linelist(linelist, wavelengths):
    d = np.loadtxt(linelist)
    w, ew = d[:, 0], d[:, -1]
    w = np.array(map(lambda x: round(x, 2), w))
    s = np.zeros(len(wavelengths))
    i = 0
    for wavelength in wavelengths:
        idx = wavelength == w
        if sum(idx):  # found the wavelength
            s[i] = ew[idx][0]
            i += 1
    return s.reshape(1, -1)

# These functions and the observations.py should go to the spectrum.py
def prepare_spectrum(spectrum):
    from scipy.interpolate import InterpolatedUnivariateSpline

    if not os.path.isfile(intname):
        raise IOError('The interval list is not in the correct place!')

    intervals = pd.read_csv(intname, comment='#', names=['start', 'end'], delimiter='\t')
    ranges = intervals.values

    # This is the wavelength of the synthetic spectra
    wlin = []
    for i, ri in enumerate(ranges):
        N = abs((ri[0] - ri[1])/0.1)
        x = np.linspace(ri[0]+0.1, ri[1], num=int(N-1), endpoint=False)
        wlin.append(x)
    wlin = np.array(wlin)
    w = np.hstack(wlin)

    wave, flux, d = read_obs_intervals(spectrum, ranges, snr=100)
    sl = InterpolatedUnivariateSpline(wave, flux, k=1)
    int_flux = sl(w)
    return int_flux.reshape(1, -1)
