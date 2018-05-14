import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
from observations import read_obs_intervals, read_observations
#from synthetic import read_linelist
import matplotlib.pyplot as plt
import os
from matplotlib import cm


def create_combined():
    #read synthetic fluxes
    spectra = glob('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results_int/*int.spec')
    #spectra = glob('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results_biggrid/*22500.spec')
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
        alpha  = specname.split('_')[6]
        #if vsini == '3.0':
        hdulist = fits.open('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results_int/' + specname)
        #hdulist = fits.open('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results_biggrid/' + specname)
        x = hdulist[1].data
        flux = x['flux']
        flux = flux.tolist()
        params = np.append(flux, [teff, logg, feh, alpha, vmic, vmac, vsini, specname])
        params = params.tolist()
        data.append(params)
        #else:
        #    pass

    hdulist = fits.open('/home/paranoia/mex/APOGEE_dwarfs_synthetic/results_int/' + specname)
    x = hdulist[1].data
    wave = x['wavelength']
    columns = np.append(wave, ['teff', 'logg', 'feh', 'alpha', 'vmic', 'vmac', 'vsini', 'spectrum'])
    header = columns.tolist()
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = header
    df.to_hdf('combined_spec.hdf', key='spectrum', mode='w')
    return


def snr_apogee(fname):
    hdulist = fits.open(fname)
    header = hdulist[1].header
    try:
        return header['SNR']
    except KeyError as e:
        return 100


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


def prepare_spectrum(spectrum, continuum=None):
    hdulist = fits.open(spectrum)
    x = hdulist[1].data
    flux = x['flux']
    if continuum is not None:
        wave = x['wavelength']
        c = np.array(continuum)
        wave = np.round(wave, 7)
        wave = wave.astype(np.str)
        ind = np.isin(wave, c, invert=True)
        flux = flux[np.where(ind)]
    return flux.reshape(1, -1)


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def plot_comparison_synthetic(df, model='linear'):
    # Model is just for saving the plots
    #teff
    x = [4000, 7000]
    y = [0, 0]
    plt.xlabel(r'$T_{eff}$ Literature (K)')
    plt.ylabel(r'$T_{eff}$ - Literature')
    plt.plot(x, y, color='black')
    #axes = plt.gca()
    #axes.set_xlim([4000, 7000])
    #axes.set_ylim([-500, 500])
    plt.grid(True)
    plt.plot(df['teff_lit'].astype(float), df['teff'].astype(float) - df['teff_lit'].astype(float), 'o', color='green')
    diff = df['teff'].astype(float) - df['teff_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(4100, 400, 'mean: %s, median: %s K' % (int(mean), int(median)))
    plt.text(4100, 350, 'std: %s, MAD %s K' % (int(std), int(mad)))
    plt.legend(frameon=False, numpoints=1)
    plt.savefig('teff_testset_' + model + '.png')
    plt.show()

    #logg
    x = [3.9, 5.0]
    y = [0, 0]
    plt.xlabel('logg Literature (dex)')
    plt.ylabel('logg - Literature')
    plt.scatter(df['logg_lit'].astype(float), df['logg'].astype(float) - df['logg_lit'].astype(float), c=df['teff_lit'].astype(float), cmap=cm.jet)
    plt.plot(x, y, color='black')
    plt.colorbar()
    diff = df['logg_lit'].astype(float) - df['logg'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(3.95, -0.1, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(3.95, -0.2, 'std: %s, MAD %s dex' % (std, mad))
    #axes = plt.gca()
    #axes.set_xlim([3.9, 5.0])
    #axes.set_ylim([-0.3, 0.3])
    plt.grid(True)
    plt.savefig('logg_testset_' + model + '.png')
    plt.show()

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
    plt.plot(df['feh_lit'].astype(float), df['[M/H]'].astype(float) - df['feh_lit'].astype(float), 'o')
    diff = df['[M/H]'].astype(float) - df['feh_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(-0.95, 0.15, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(-0.95, 0.10, 'std: %s, MAD %s dex' % (std, mad))
    plt.savefig('metal_testset_' + model + '.png')
    plt.show()

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
    plt.plot(df['alpha_lit'].astype(float), df['alpha'].astype(float) - df['alpha_lit'].astype(float), 'o')
    diff = df['alpha'].astype(float) - df['alpha_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(-0.05, 0.3, 'mean: %s, median: %s dex' % (mean, median))
    plt.text(-0.05, 0.25, 'std: %s, MAD %s dex' % (std, mad))
    plt.savefig('alpha_testset_' + model + '.png')
    plt.show()
    return


def save_and_compare_synthetic(d, model='linear'):
    #Model is just for naming the plots
    df_ml = pd.DataFrame(data=d)
    # Save ML parameters
    df_ml.to_csv('results_ML.dat', sep='\t')
    #read synthetic fluxes
    spectra = df_ml['specname'].as_matrix()
    spectra = list(map(lambda x: x.split('/')[-1], spectra))

    data = []
    for specname in spectra[:]:
        teff  = specname.split('_')[0]
        logg  = specname.split('_')[1]
        feh   = specname.split('_')[2]
        vmic  = specname.split('_')[3]
        vmac  = specname.split('_')[4]
        vsini = specname.split('_')[5]
        alpha  = specname.split('_')[6]
        params = [teff, logg, feh, alpha, vmic, vmac, vsini, specname]
        data.append(params)

    columns = ['teff_lit', 'logg_lit', 'feh_lit', 'alpha_lit', 'vmic_lit', 'vmac_lit', 'vsini_lit', 'specname']
    data = np.array(data)
    df = pd.DataFrame(data)
    df.columns = columns
    comp = pd.merge(df, df_ml, how='left', on=['specname'])
    comp.to_csv('comparison_ML.dat', sep='\t', na_rep='nan')
    plot_comparison_synthetic(comp, model)
    return
