import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.interpolate import InterpolatedUnivariateSpline

def create_combined():
    #read synthetic fluxes
    path_of_grid = '/home/mtsantaki/oporto/gaia_synthetic_kurucz/results_005/'
    spectra = glob(path_of_grid + '*11200.spec')
    spectra = list(map(lambda x: x.split('/')[-1], spectra))

    data = []
    for i, specname in enumerate(spectra[:]):
        print(i)
        teff  = specname.split('_')[0]
        logg  = specname.split('_')[1]
        feh   = specname.split('_')[2]
        vmic  = specname.split('_')[3]
        vmac  = specname.split('_')[4]
        vsini = specname.split('_')[5]
        alpha = specname.split('_')[6]
        #if vsini == '3.0':
        hdulist = fits.open(path_of_grid + specname)
        x = hdulist[1].data
        flux = x['flux']
        flux = flux.tolist()
        params = np.append(flux, [teff, logg, feh, alpha, vmic, vmac, vsini])
        params = params.tolist()
        data.append(params)
        #else:
        #    pass

    hdulist = fits.open(path_of_grid + specname)
    x = hdulist[1].data
    wave = x['wavelength']
    wave = np.round(wave, 2)
    columns = np.append(wave, ['teff', 'logg', 'feh', 'alpha', 'vmic', 'vmac', 'vsini'])
    header = columns.tolist()
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = header
    print('Writing to file..')
    df.to_hdf('spec_ML_kurucz.hdf', key='df', mode='w')
    return

def create_combined_sn4():
    #read synthetic fluxes
    sn4 = pd.read_csv('sn4/params_sn4.dat', comment='#', skiprows=1, delimiter=r'\s+', usecols=(0,1,3,8),
    names=['specname', 'teff_lit', 'logg_lit', 'metal_lit'], converters={'specname': lambda x : 'HIP' + x + '.spec'})
    sn4.dropna(inplace=True)

    data = []
    for i, spec in enumerate(sn4.specname.values):
        print(i, spec)
        hdulist = fits.open('sn4/' + spec)
        x = hdulist[1].data
        flux = x['flux']
        flux = flux.tolist()
        alpha, vmic, vmac, vsini = (0, 0, 0, 0)
        params = np.append(flux, [sn4.teff_lit.values[i], sn4.logg_lit.values[i], sn4.metal_lit.values[i], alpha, vmic, vmac, vsini])
        params = params.tolist()
        data.append(params)
        #else:
        #    pass

    hdulist = fits.open('sn4/' + spec)
    x = hdulist[1].data
    wave = x['wavelength']
    wave = np.round(wave, 2)
    columns = np.append(wave, ['teff', 'logg', 'feh', 'alpha', 'vmic', 'vmac', 'vsini'])
    header = columns.tolist()
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = header
    print('Writing to file..')
    df.to_hdf('spec_ML_sn4.hdf', key='df', mode='w')
    return

def meanstdv(x):
    '''Simple statistics'''
    x = x[~np.isnan(x)]
    mean   = np.mean(x)
    median = np.median(x)
    mad    = np.median(np.absolute(x - np.median(x)))
    std    = np.std(x, ddof=1)
    stderr = std / np.sqrt(len(x))
    return round(mean, 3), round(median, 3), round(std, 3), round(mad, 3)

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def save_and_compare_synthetic(d):

    columns = ['teff_lit', 'logg_lit', 'metal_lit', 'alpha_lit', 'teff', 'logg', 'metal', 'alpha']
    comp = pd.DataFrame(data=d.T, columns=columns)
    #comp = comp[comp['logg_lit'] > 3.9]
    #comp = comp[comp['metal_lit'] > -1.9]
    #results = []
    label = ['teff', 'logg', 'metal', 'alpha']
    plt.scatter(comp['teff_lit'].astype(float), comp['teff'].astype(float) - comp['teff_lit'].astype(float), c=comp['metal_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([4000, 6700], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('teff_linear_test.png')
    plt.show()

    plt.scatter(comp['logg_lit'].astype(float), comp['logg'].astype(float) - comp['logg_lit'].astype(float), c=comp['teff_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([1.5, 5.0], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('logg_linear_test.png')
    plt.show()

    plt.scatter(comp['metal_lit'].astype(float), comp['metal'].astype(float) - comp['metal_lit'].astype(float), c=comp['teff_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([-2.0, 0.6], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('metal_linear_test.png')
    plt.show()

    plt.scatter(comp['alpha_lit'].astype(float), comp['alpha'].astype(float) - comp['alpha_lit'].astype(float), c=comp['teff_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([-0.0, 0.4], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('alpha_linear_test.png')
    plt.show()

    for l in label:
    #    plt.figure()
    #    plt.scatter(comp[l+'_lit'].astype(float), comp[l].astype(float) - comp[l+'_lit'].astype(float), s=40, alpha=0.5, color='green', label=str(l))
        diff = comp[l].astype(float) - comp[l+'_lit'].astype(float)
        r = meanstdv(diff)
        print('%s: mean = %s, median = %s, std = %s, mad = %s' % (l, r[0], r[1], r[2], r[3]))
    #    plt.legend(frameon=False, numpoints=1)
    #    plt.xlabel(str(l) + ' synthetic')
    #    plt.grid(True)
        #plt.savefig(l + '_linear_PowerTransformer' + '.png')
        #plt.show()
    return

def save_and_compare_sn4(d):

    columns = ['teff_lit', 'logg_lit', 'metal_lit', 'teff', 'logg', 'metal']
    comp = pd.DataFrame(data=d.T, columns=columns)
    #comp = comp[comp['logg_lit'] > 3.9]
    #comp = comp[comp['metal_lit'] > -1.9]
    #results = []
    label = ['teff', 'logg', 'metal']
    plt.scatter(comp['teff_lit'].astype(float), comp['teff'].astype(float) - comp['teff_lit'].astype(float), c=comp['metal_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([4000, 6700], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('teff_linear_test.png')
    plt.show()

    plt.scatter(comp['logg_lit'].astype(float), comp['logg'].astype(float) - comp['logg_lit'].astype(float), c=comp['teff_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([1.5, 5.0], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('logg_linear_test.png')
    plt.show()

    plt.scatter(comp['metal_lit'].astype(float), comp['metal'].astype(float) - comp['metal_lit'].astype(float), c=comp['teff_lit'], alpha=0.8, cmap=cm.jet)
    plt.plot([-2.0, 0.6], [0.0, 0.0], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.grid()
    #plt.savefig('metal_linear_test.png')
    plt.show()

    for l in label:
    #    plt.figure()
    #    plt.scatter(comp[l+'_lit'].astype(float), comp[l].astype(float) - comp[l+'_lit'].astype(float), s=40, alpha=0.5, color='green', label=str(l))
        diff = comp[l].astype(float) - comp[l+'_lit'].astype(float)
        r = meanstdv(diff)
        print('%s: mean = %s, median = %s, std = %s, mad = %s' % (l, r[0], r[1], r[2], r[3]))
    #    plt.legend(frameon=False, numpoints=1)
    #    plt.xlabel(str(l) + ' synthetic')
    #    plt.grid(True)
        #plt.savefig(l + '_linear_PowerTransformer' + '.png')
        #plt.show()
    return

if __name__ == '__main__':
    create_combined_sn4()
