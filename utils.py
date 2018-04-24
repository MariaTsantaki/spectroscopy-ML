import numpy as np
import pandas as pd
from glob import glob


def create_combined():

    from astropy.io import fits

    #read synthetic fluxes
    spectra = glob('synthetic_spec/results/*.spec')
    spectra = list(map(lambda x: x[23:], spectra))

    data = []
    for specname in spectra[:]:
        hdulist = fits.open('synthetic_spec/results/' + specname)
        x = hdulist[1].data
        flux = x['flux']
        wave = x['wavelength']
        teff  = specname.split('_')[0]
        logg  = specname.split('_')[1]
        feh   = specname.split('_')[2]
        vmic  = specname.split('_')[3]
        vmac  = specname.split('_')[4]
        vsini = specname.split('_')[5]
        #flux = flux.tolist()
        #wave = wave.tolist()
        #params = flux.extend([teff, logg, feh, vmic, vmac, vsini])
        #params = np.insert(flux, len(flux), [teff, logg, feh, vmic, vmac, vsini, specname])
        params = np.append(flux, [teff, logg, feh, vmic, vmac, vsini, specname])
        params = params.tolist()
        data.append(params)
        #columns = wave.append(['teff', 'logg', 'feh', 'vmic', 'vmac', 'vsini'])
        #params = np.insert(flux, len(flux), [teff, logg, feh, vmic, vmac, vsini])
        columns = np.append(wave, ['teff', 'logg', 'feh', 'vmic', 'vmac', 'vsini', 'spectrum'])

    header = columns.tolist()
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = header
    df.to_csv('combined_spec.csv')
    return


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

#create_combined()
