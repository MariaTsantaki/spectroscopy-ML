import numpy as np
import pandas as pd
from glob import glob
from astropy.io import fits
from observations import read_obs_intervals, read_observations, snr_apogee
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
    print('Writing to file..')
    df.to_csv('data/combined_spec.hdf')
    return


def prepare_spectrum_synth(spectrum, continuum=None):
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
        wave = wave[np.where(ind)]
    return flux.reshape(1, -1), wave


def prepare_spectrum(spectrum, continuum=None, intname='data/intervals.lst'):
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
    print('mean %s median %s std %s stderr %s mad %s' % (round(mean, 3), round(median, 3), round(std, 3), round(stderr, 3), round(mad, 3)))
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
    axes = plt.gca()
    axes.set_xlim([4000, 7000])
    axes.set_ylim([-1500, 1500])
    plt.grid(True)
    plt.plot(df['teff_lit'].astype(float), df['teff'].astype(float) - df['teff_lit'].astype(float), 'o', color='green', label='fasma')
    plt.plot(df['teff_lit'].astype(float), df['teff_calib'].astype(float) - df['teff_lit'].astype(float), 'o', color='red', label='APOGEE')
    diff = df['teff'].astype(float) - df['teff_lit'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    #plt.text(4100, 400, 'mean: %s, median: %s K' % (int(mean), int(median)))
    #plt.text(4100, 350, 'std: %s, MAD %s K' % (int(std), int(mad)))
    diff = df['teff_calib'].astype(float) - df['teff_lit'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    #plt.text(4100, 300, 'mean: %s, median: %s K' % (int(mean), int(median)))
    #plt.text(4100, 250, 'std: %s, MAD %s K' % (int(std), int(mad)))
    plt.legend(frameon=False, numpoints=1)
    plt.savefig('teff_apogee.png')
    plt.show()

    #logg
    x = [3.2, 5.0]
    y = [0, 0]
    plt.xlabel('logg trigonometric (dex)')
    plt.ylabel('logg - Literature')
    #yerror = np.sqrt(df['erlogg'].astype(float)**2 + df['erloggp'].astype(float)**2)
    #yerror = pd.Series(yerror).values
    plt.plot(df['logg_lit'].astype(float), df['logg'].astype(float) - df['logg_lit'].astype(float), 'o', label='fasma')
    plt.plot(df['logg_lit'].astype(float), df['logg_uncalib'].astype(float) - df['logg_lit'].astype(float), 'o', label='APOGEE uncalib')
    plt.plot(df['logg_lit'].astype(float), df['logg_lit'].astype(float) - df['loggp'].astype(float), 'o', label='lit - paral')
    #plt.plot(df['logg_lit'].astype(float), df['logg'].astype(float) - df['logg_lit'].astype(float), 'o', label='fasma')
    #plt.plot(df['logg_lit'].astype(float), df['logg_uncalib'].astype(float) - df['logg_lit'].astype(float), 'o', color='black', label='APOGEE uncalib')
    plt.plot(x, y, color='black')
    #plt.colorbar()
    plt.legend(frameon=False, numpoints=1)
    diff = df['logg_lit'].astype(float) - df['logg'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    #plt.text(3.3, -0.5, 'mean: %s, median: %s dex' % (mean, median))
    #plt.text(3.3, -0.6, 'std: %s, MAD %s dex' % (std, mad))
    diff = df['logg_uncalib'].astype(float) - df['logg'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    #plt.text(3.3, -0.7, 'mean: %s, median: %s dex' % (mean, median))
    #plt.text(3.3, -0.8, 'std: %s, MAD %s dex' % (std, mad))
    axes = plt.gca()
    axes.set_xlim([3.2, 5.0])
    axes.set_ylim([-2, 2])
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
    #yerror = np.sqrt(df['erfeh'].astype(float)**2 + df['erfeh_lit'].astype(float)**2)
    #yerror = pd.Series(yerror).values
    plt.plot(df['feh_lit'].astype(float), df['metal'].astype(float) - df['feh_lit'].astype(float), 'o', label='fasma')
    plt.plot(df['feh_lit'].astype(float), df['feh_calib'].astype(float) - df['feh_lit'].astype(float), 'o', label='APOGEE calib')
    diff = df['metal'].astype(float) - df['feh_lit'].astype(float)
    print('ML-LIT')
    #mean, median, std, mad = meanstdv(diff)
    results.append(meanstdv(diff))
    #plt.text(-0.95, 0.3, 'mean: %s, median: %s dex' % (mean, median))
    #plt.text(-0.95, 0.25, 'std: %s, MAD %s dex' % (std, mad))
    diff = df['feh_calib'].astype(float) - df['feh_lit'].astype(float)
    print('APOGEE-LIT')
    mean, median, std, mad = meanstdv(diff)
    #plt.text(-0.95, 0.2, 'mean: %s, median: %s dex' % (mean, median))
    #plt.text(-0.95, 0.15, 'std: %s, MAD %s dex' % (std, mad))
    plt.legend(frameon=False, numpoints=1)
    plt.savefig('metal_apogee.png')
    plt.show()
    results.append((0.0, 0.0, 0.0, 0.0))
    return results


def plot_comparison_synthetic(df, model):

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
    plt.scatter(df['teff_lit'].astype(float), df['teff'].astype(float) - df['teff_lit'].astype(float), s=40, alpha=0.5, color='green')
    diff = df['teff'].astype(float) - df['teff_lit'].astype(float)
    mean, median, std, stderr, mad = meanstdv(diff)
    plt.text(4100, 400, 'mean: %s, median: %s K' % (int(mean), int(median)))
    plt.text(4100, 350, 'std: %s, MAD %s K' % (int(std), int(mad)))
    #plt.legend(frameon=False, numpoints=1)
    plt.savefig('teff_testset_' + model + '.png')
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
    plt.savefig('logg_testset_' + model + '.png')
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
    plt.savefig('metal_testset_' + model + '.png')
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
    plt.savefig('alpha_testset_' + model + '.png')
    #plt.show()

    return


def save_and_compare_synthetic(d, model):

    df_ml = pd.DataFrame(data=d)
    # Save ML parameters
    #df_ml.to_csv('results_ML.dat', sep='\t')
    #read synthetic fluxes
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
    #comp.to_csv('comparison_ML.dat', sep='\t', na_rep='nan')

    label = ['teff', 'logg', 'metal', 'alpha']
    results = []
    for l in label:
        plt.figure()
        plt.scatter(comp[l+'_lit'].astype(float), comp[l].astype(float) - comp[l+'_lit'].astype(float), s=40, alpha=0.5, color='green', label=str(l))
        diff = comp[l].astype(float) - comp[l+'_lit'].astype(float)
        #mean, median, std, stderr, mad = meanstdv(diff)
        results.append(meanstdv(diff))
        plt.legend(frameon=False, numpoints=1)
        plt.grid(True)
        #plt.savefig(l + '_testset_' + model + '.png')
        plt.show()
    return results


def save_and_compare_apogee(d, model):

    df_ml = pd.DataFrame(data=d)
    # Save ML parameters
    #df_ml.to_csv('results_ML.dat', sep='\t')
    # Compare with APOGEE values
    #column_names = ['[M/H]', 'alpha', 'logg', 'specname', 'teff']
    #df_ml.columns = column_names
    #print(df_ml)
    # Compare with APOGEE values
    df_ap = pd.read_csv('apogee_params.dat', sep='\t')
    #print(df_ap['specname'])
    comp = pd.merge(df_ap, df_ml, how='left', on=['specname'])
    comp.replace(to_replace=[-9999], value='nan', inplace=True)
    #comp.to_csv('comparison_ML.dat', sep='\t', na_rep='nan')
    results = plot_comparison_apogee(comp)
    #label = ['teff', 'logg', 'metal', 'alpha']
    #for l in label:
    #    diff = comp[l].astype(float) - comp[l+'_lit'].astype(float)
    #    diff = diff[diff > -8000]
    #    diff.dropna(inplace=True)
    #    results.append(meanstdv(diff))
    return results
