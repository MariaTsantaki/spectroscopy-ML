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
def read_linelist(fname='moog_apogee_atom_cn_clean.lst', intname='intervals.lst'):
    '''Read the line list return atomic data and ranges

    Input
    -----
    fname : str
      File that contains the linelist
    intname : str
      File that contains the intervals

    Output
    ------
    ranges : wavelength ranges of the linelist
    atomic : atomic data
    '''

    if not os.path.isfile(intname):
        raise IOError('The interval list is not in the correct place!')
    if not os.path.isfile(fname):
        raise IOError('The line list is not in the correct place!')
    lines = pd.read_csv(fname, skiprows=1, comment='#', delimiter=r'\s+', usecols=range(6),
    names=['wl', 'elem', 'excit', 'loggf', 'vdwaals', 'Do'],
    converters={'Do': lambda x : x.replace("nan"," "), 'vdwaals': lambda x : float(x)})
    lines.sort_values(by='wl', inplace=True)

    intervals = pd.read_csv(intname, comment='#', names=['start', 'end'], delimiter='\t')
    ranges = intervals.values
    atomic = []
    for i, ri in enumerate(intervals.values):
        a = lines[(lines.wl>ri[0]) & (lines.wl<ri[1])]
        a = a.as_matrix()
        atomic.append(a)
    atomic = np.vstack(atomic)
    N = len(atomic)
    print('Linelist contains %s lines in %s intervals' % (N, len(ranges)))

    # Create line list for MOOG
    fmt = ["%9s", "%8s", "%9s", "%9s", "%6s", "%8s"]
    header = 'Wavelength     ele       EP      loggf   vdwaals   Do'
    np.savetxt('linelist.moog', atomic, fmt=fmt, header=header)
    return ranges, atomic

def prepare_spectrum(spectrum):
    from scipy.interpolate import InterpolatedUnivariateSpline

    r, a = read_linelist('moog_apogee_atom_cn_clean_short.lst', intname='intervals.lst')
    #wl = np.array([])
    #for i in np.arange(8575): wl = np.append(wl, 4.179 + 6.E-6*i)
    #wl = 10.**(wl)

    #w = []
    #for i, ri in enumerate(r):
    #    wll = wl[np.where((wl >= ri[0]) & (wl <= ri[1]))]
    #    w.append(wll)

    #w = np.array(w)
    # This is something weird :)
    hdulist = fits.open('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results/6800_4.381_0.28_1.75_7.81_2.0_0.0_22500.spec')
    x = hdulist[1].data
    w = x['wavelength']
    f = x['flux']

    wave, flux, d = read_obs_intervals(spectrum, r, snr=100)
    sl = InterpolatedUnivariateSpline(wave, flux, k=1)
    int_flux = sl(w)
    return int_flux.reshape(1, -1)
