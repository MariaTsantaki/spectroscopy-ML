import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
from observations import read_obs_intervals, read_observations, snr_apogee
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.interpolate import InterpolatedUnivariateSpline

def create_combined():
    #read synthetic fluxes
    path_of_grid = 'data/train_data/'
    spectra = glob(path_of_grid + '*11200_int.spec')
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
    columns = np.append(wave, ['teff', 'logg', 'feh', 'alpha', 'vmic', 'vmac', 'vsini'])
    header = columns.tolist()
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = header
    print('Writing to file..')
    df.to_hdf('spec_ML.hdf', key='df', mode='w')
    return

def int_spectrum_synth(spectrum, continuum=None):
    hdulist = fits.open(spectrum)
    x = hdulist[1].data
    flux = x['flux']
    if continuum is not None:
        wave = x['wavelength']
        w = continuum.astype(np.float)
        sl = InterpolatedUnivariateSpline(wave, flux, k=1)
        flux = sl(w)
    else:
        wave = x['wavelength']
    return flux.reshape(1, -1), wave

def prepare_spectrum_synth(spectrum, continuum=None):
    hdulist = fits.open(spectrum)
    x = hdulist[1].data
    flux = x['flux']
    if continuum is not None:
        continuum = continuum.astype(np.float)
        continuum = np.round(continuum, 7)
        continuum = continuum.astype(np.str)
        wave = x['wavelength']
        wave = np.round(wave, 7)
        wave = wave.astype(np.str)
        ind = np.isin(wave, continuum)
        flux = flux[np.where(ind)]
        wave = wave[np.where(ind)]
    else:
        wave = x['wavelength']
    return flux.reshape(1, -1), wave

def prepare_spectrum(spectrum, continuum=None, intname='intervals.lst'):
    intervals = pd.read_csv('rawLinelist/%s' % intname, comment='#', names=['start', 'end'], delimiter='\t')
    ranges = intervals.values

    snr = snr_apogee(spectrum)
    wave, flux, d = read_obs_intervals(spectrum, ranges, snr)

    if continuum is not None:
        c = np.array(continuum)
        c = c.astype(np.float)
        c = np.round(c, 7)
        wave = np.round(wave, 7)
        wave = wave.astype(np.str)
        ind = np.isin(wave, c, invert=True)
        flux = flux[np.where(ind)]
        wave = wave[np.where(ind)]
    return flux.reshape(1, -1), wave

def meanstdv(x):
    '''Simple statistics'''
    x = x[~np.isnan(x)]
    mad    = np.median(np.absolute(x - np.median(x)))
    mean   = np.mean(x)
    median = np.median(x)
    std    = np.std(x, ddof=1)
    stderr = std / np.sqrt(len(x))
    return round(mean, 3), round(median, 3), round(std, 3), round(mad, 3)

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def plot_comparison_apogee(df):

    results = []
    #teff
    x = [4000, 7000]
    y = [0, 0]
    plt.xlabel(r'$T_{eff}$ Literature (K)')
    plt.ylabel(r'$T_{eff}$ - Literature')
    plt.plot(x, y, color='black')
    plt.grid(True)
    plt.plot(df['teff_lit'].astype(float), df['teff'].astype(float) - df['teff_lit'].astype(float), 'o', color='green', label='fasma')
    plt.plot(df['teff_lit'].astype(float), df['teff_calib'].astype(float) - df['teff_lit'].astype(float), 'o', color='red', label='APOGEE')
    diff = df['teff'].astype(float) - df['teff_lit'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    diff = df['teff_calib'].astype(float) - df['teff_lit'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    plt.legend(frameon=False, numpoints=1)
    plt.savefig('teff_apogee.png')
    plt.show()

    #logg
    x = [3.2, 5.0]
    y = [0, 0]
    plt.xlabel('logg trigonometric (dex)')
    plt.ylabel('logg - Literature')
    plt.plot(df['logg_lit'].astype(float), df['logg'].astype(float) - df['logg_lit'].astype(float), 'o', label='fasma')
    plt.plot(df['logg_lit'].astype(float), df['logg_uncalib'].astype(float) - df['logg_lit'].astype(float), 'o', label='APOGEE uncalib')
    plt.plot(df['logg_lit'].astype(float), df['logg_lit'].astype(float) - df['loggp'].astype(float), 'o', label='lit - paral')
    plt.plot(x, y, color='black')
    #plt.colorbar()
    plt.legend(frameon=False, numpoints=1)
    diff = df['logg_lit'].astype(float) - df['logg'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    diff = df['logg_uncalib'].astype(float) - df['logg'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    plt.grid(True)
    plt.savefig('logg_apogee.png')
    plt.show()

    #feh
    x = [-2.0, 0.6]
    y = [0, 0]
    plt.xlabel('[Fe/H] Literature (dex)')
    plt.ylabel('[Fe/H] - Literature')
    plt.plot(x, y, color='black')
    axes = plt.gca()
    axes.set_xlim([-1.0, 0.6])
    axes.set_ylim([-1.0, 1.0])
    plt.grid(True)
    plt.plot(df['feh_lit'].astype(float), df['metal'].astype(float) - df['feh_lit'].astype(float), 'o', label='fasma')
    plt.plot(df['feh_lit'].astype(float), df['feh_calib'].astype(float) - df['feh_lit'].astype(float), 'o', label='APOGEE calib')
    diff = df['metal'].astype(float) - df['feh_lit'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    diff = df['feh_calib'].astype(float) - df['feh_lit'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    plt.legend(frameon=False, numpoints=1)
    plt.savefig('metal_apogee.png')
    plt.show()
    results.append((0.0, 0.0, 0.0, 0.0))
    return results

def plot_comparison_synthetic(df, class_name='linear'):

    #teff
    x = [4000, 7000]
    y = [0, 0]
    plt.xlabel(r'$T_{eff}$ Literature (K)')
    plt.ylabel(r'$T_{eff}$ - Literature')
    plt.plot(x, y, color='black')
    plt.grid(True)
    plt.scatter(df['teff_lit'].astype(float), df['teff'].astype(float) - df['teff_lit'].astype(float), s=40, alpha=0.5, color='green')
    diff = df['teff'].astype(float) - df['teff_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(4100, 400, 'mean: %s, median: %s K' % (int(mean), int(median)))
    plt.text(4100, 350, 'std: %s, MAD %s K' % (int(std), int(mad)))
    #plt.legend(frameon=False, numpoints=1)
    plt.savefig('teff_testset_' + class_name + '.png')
    #plt.show()

    #logg
    x = [3.9, 5.0]
    y = [0, 0]
    plt.xlabel('logg Literature (dex)')
    plt.ylabel('logg - Literature')
    plt.scatter(df['logg_lit'].astype(float), df['logg'].astype(float) - df['logg_lit'].astype(float), c=df['teff_lit'].astype(float), cmap=cm.jet)
    plt.plot(x, y, color='black')
    plt.colorbar()
    #plt.legend()
    diff = df['logg_lit'].astype(float) - df['logg'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(3.95, -0.1, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(3.95, -0.2, 'std: %s, MAD %s dex' % (std, mad))
    #axes = plt.gca()
    #axes.set_xlim([3.9, 5.0])
    #axes.set_ylim([-0.3, 0.3])
    plt.grid(True)
    plt.savefig('logg_testset_' + class_name + '.png')
    #plt.show()

    #feh
    x = [-2.0, 0.5]
    y = [0, 0]
    plt.xlabel('[Fe/H] Literature (dex)')
    plt.ylabel('[Fe/H] - Literature')
    plt.plot(x, y, color='black')
    #axes = plt.gca()
    #axes.set_xlim([-2.0, 0.5])
    #axes.set_ylim([-0.2, 0.2])
    plt.grid(True)
    plt.scatter(df['feh_lit'].astype(float), df['[M/H]'].astype(float) - df['feh_lit'].astype(float), alpha=0.5, s=40)
    diff = df['[M/H]'].astype(float) - df['feh_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(-0.95, 0.15, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(-0.95, 0.10, 'std: %s, MAD %s dex' % (std, mad))
    #plt.legend(frameon=False, numpoints=1)
    plt.savefig('metal_testset_' + class_name + '.png')
    #plt.show()

    #alpha
    x = [-0.1, 0.5]
    y = [0, 0]
    plt.xlabel('alpha Literature (dex)')
    plt.ylabel('alpha - Literature')
    plt.plot(x, y, color='black')
    #axes = plt.gca()
    #axes.set_xlim([-0.1, 0.5])
    #axes.set_ylim([-0.15, 0.15])
    plt.grid(True)
    plt.scatter(df['alpha_lit'].astype(float), df['alpha'].astype(float) - df['alpha_lit'].astype(float), s=40, alpha=0.5)
    diff = df['alpha'].astype(float) - df['alpha_lit'].astype(float)
    #diff = diff[diff > -8000]
    #print(len(diff))
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(-0.05, 0.3, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(-0.05, 0.25, 'std: %s, MAD %s dex' % (std, mad))
    #plt.legend(frameon=False, numpoints=1)
    plt.savefig('alpha_testset_' + class_name + '.png')
    #plt.show()

    return

def save_and_compare_synthetic(d, class_name='linear'):

    df_ml = pd.DataFrame(data=d)
    spectra = df_ml['specname'].values
    spectra = list(map(lambda x: x.split('/')[-1], spectra))

    data = []
    for specname in spectra[:]:
        teff  = specname.split('_')[0]
        logg  = specname.split('_')[1]
        metal = specname.split('_')[2]
        vmic  = specname.split('_')[3]
        vmac  = specname.split('_')[4]
        vsini = specname.split('_')[5]
        alpha  = specname.split('_')[6]
        params = [teff, logg, metal, alpha, vmic, vmac, vsini, specname]
        data.append(params)

    columns = ['teff_lit', 'logg_lit', 'metal_lit', 'alpha_lit', 'vmic_lit', 'vmac_lit', 'vsini_lit', 'specname']
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = columns
    comp = pd.merge(df, df_ml, how='left', on=['specname'])

    label = ['teff', 'logg', 'metal', 'alpha']
    results = []
    for l in label:
        plt.figure()
        plt.scatter(comp[l+'_lit'].astype(float), comp[l].astype(float) - comp[l+'_lit'].astype(float), s=40, alpha=0.5, color='green', label=str(l))
        diff = comp[l].astype(float) - comp[l+'_lit'].astype(float)
        r = meanstdv(diff)
        results.append([r[0], r[1], r[2], r[3]])
        print('%s: mean = %s, median = %s, std = %s, mad = %s' % (l, r[0], r[1], r[2], r[3]))
        plt.legend(frameon=False, numpoints=1)
        plt.xlabel(str(l) + ' synthetic')
        plt.grid(True)
        #plt.savefig(l + '_synthetic_' + class_name + '.png')
        plt.show()
    return results

def save_and_compare_apogee(d, model):

    df_ml = pd.DataFrame(data=d)
    # Compare with APOGEE values
    df_ap = pd.read_csv('apogee_params.dat', sep='\t')
    comp = pd.merge(df_ap, df_ml, how='left', on=['specname'])
    comp.replace(to_replace=[-9999], value='nan', inplace=True)
    results = plot_comparison_apogee(comp)
    return results

if __name__ == '__main__':
    create_combined()
