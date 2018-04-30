# -*- coding: utf8 -*-

# My imports
from __future__ import division
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from PyAstronomy import pyasl


def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)


def local_norm(obs_fname, r, SNR, method='linear', plot=False):
    '''Local Normalisation function. Make a linear fit from the maximum points
    of each segment.
    Input
    -----
    obs_fname : observations file
    r : range of the interval

    Output
    ------
    new_flux : normalized flux
    '''

    # Define the area of Normalization
    start_norm = r[0] - 1.0
    end_norm = r[1] + 1.0

    #Transform SNR to noise
    if SNR is None:
        noise = 0.0
    else:
        SNR = float(SNR)
        noise = 1.0/(SNR)
    #Read observations
    wave_obs, flux_obs, delta_l = read_observations(obs_fname, start_norm, end_norm)

    # Clean for cosmic rays
    med = np.median(flux_obs)
    ind0 = np.where(flux_obs == 0)
    flux_obs[flux_obs == 0] = med
    med = np.median(flux_obs)
    sigma = mad(flux_obs)
    n = len(flux_obs)
    fluxout = np.zeros(n)
    for i in range(n):
        if flux_obs[i] > (med + (sigma*5.0)):
           fluxout[i] = med
	else:
	   fluxout[i] = flux_obs[i]
    flux_obs = fluxout

    if (end_norm-start_norm) > 40:
        #y = np.array_split(flux_obs, 4)
        #x = np.array_split(wave_obs, 4)
        y = np.array_split(flux_obs, 3)
        x = np.array_split(wave_obs, 3)
        index_max1 = np.sort(np.argsort(y[0])[-11:-3])  # this can be done better
        index_max2 = np.sort(np.argsort(y[1])[-11:-3])  # this can be done better
        index_max3 = np.sort(np.argsort(y[2])[-11:-3])  # this can be done better
        #index_max4 = np.sort(np.argsort(y[3])[-11:-3])  # this can be done better
        f_max1 = y[0][index_max1]
        f_max2 = y[1][index_max2]
        f_max3 = y[2][index_max3]
        #f_max4 = y[3][index_max4]
        w_max1 = x[0][index_max1]
        w_max2 = x[1][index_max2]
        w_max3 = x[2][index_max3]
        #w_max4 = x[3][index_max4]
        #f_max = np.concatenate(((f_max1, f_max2, f_max3, f_max4)))
        #w_max = np.concatenate(((w_max1, w_max2, w_max3, w_max4)))
        f_max = np.concatenate((f_max1, f_max2, f_max3))
        w_max = np.concatenate((w_max1, w_max2, w_max3))

    else:
        # Divide in 2 and find the maximum points
        y = np.array_split(flux_obs, 2)
        x = np.array_split(wave_obs, 2)
        index_max1 = np.sort(np.argsort(y[0])[-11:-3])  # this can be done better
        index_max2 = np.sort(np.argsort(y[1])[-11:-3])  # this can be done better
        f_max1 = y[0][index_max1]
        f_max2 = y[1][index_max2]
        w_max1 = x[0][index_max1]
        w_max2 = x[1][index_max2]
        f_max = np.concatenate((f_max1, f_max2))
        w_max = np.concatenate((w_max1, w_max2))

    if method == 'scalar':
        # Divide with the median of maximum values.
        new_flux = flux_obs/np.median(f_max)
        if snr<20:
            new_flux =  new_flux + (2.0*noise)
        elif 20<=snr<200:
            new_flux =  new_flux + (1.5*noise)
        elif 200<=snr<350:
            new_flux =  new_flux + (1.0*noise)
        elif 350<=snr:
            new_flux =  new_flux + (0.0*noise)
    if method == 'linear':
        #z = np.polyfit(w_max, f_max, 1)
        #p = np.poly1d(z)
        #new_flux = flux_obs/p(wave_obs)
        #new_flux1 = new_flux + 1.0*noise
        z = np.polyfit(w_max, f_max-(1.0*f_max*noise), 1)
        p = np.poly1d(z)
        new_flux2 = flux_obs/p(wave_obs)

    new_flux2[ind0] = 1.0
    # Clean for cosmic rays
    med = np.median(new_flux2)
    sigma = mad(new_flux2)
    n = len(new_flux2)
    fluxout = np.zeros(n)
    for i in range(n):
        if new_flux2[i] > (med + (sigma*4.0)):
           fluxout[i] = med
	else:
	   fluxout[i] = new_flux2[i]
    new_flux2 = fluxout

    wave = wave_obs[np.where((wave_obs >= float(r[0])) & (wave_obs <= float(r[1])))]
    #new_flux1 = new_flux1[np.where((wave_obs >= float(r[0])) & (wave_obs <= float(r[1])))]
    new_flux = new_flux2[np.where((wave_obs >= float(r[0])) & (wave_obs <= float(r[1])))]

    if plot:
        plt.plot(wave_obs, flux_obs)
        y = p(wave_obs)
        plt.plot(wave_obs, y)
        plt.plot(w_max, f_max, 'o')
        plt.show()

        x = [start_norm, end_norm]
        y = [1.0, 1.0]
        plt.plot(x, y)
        plt.plot(wave, new_flux, label='1')
        #plt.plot(wave, new_flux1, label='1')
        #plt.plot(wave, new_flux2, label='0.5')
        plt.legend()
        plt.show()
    return wave, new_flux, delta_l


def read_observations(fname, start_synth, end_synth):
    """Read observed spectrum of different types and return wavelength and flux.
    Input
    -----
    fname : filename of the spectrum. Currently only fits and text files accepted.
    start_synth : starting wavelength where the observed spectrum is cut
    end_synth : ending wavelength where the observed spectrum is cut

    Output
    -----
    wavelength_obs : observed wavelength
    flux_obs : observed flux
    """
    # These are the approved formats
    extension = ('.dat', '.txt', '.spec', '.fits', '.csv')
    if fname.endswith(extension):
        if (fname[-4:] == '.dat') or (fname[-4:] == '.txt'):
            with open(fname, 'r') as f:
                lines = (line for line in f if not line[0].isalpha())  # skip header
                wave, flux = np.loadtxt(lines, unpack=True, usecols=(0, 1))

        elif fname[-4:] == '.csv':
            with open(fname, 'r') as f:
                lines = (line for line in f if not line[0].isalpha())  # skip header
                w, f = np.loadtxt(lines, delimiter=',', unpack=True, usecols=(0, 1))
                flux, wave = pyasl.dopplerShift(w, f, -0.36, edgeHandling="firstlast")
                #wave = (100000000/(wave))
                #wl = wave[::-1]
                #wave = wl / (1.0 +  (5.792105E-2/(238.0185E0 - (1.E4/wl)**2)) + (1.67917E-3/(57.362E0 - (1.E4/wl)**2)))

        elif fname[-5:] == '.fits':
            hdulist = fits.open(fname)
            header = hdulist[0].header
            # Only 1-D spectrum accepted.
            flux = hdulist[0].data  # flux data in the primary
            flux = np.array(flux, dtype=np.float64)
            start_wave = header['CRVAL1']  # initial wavelenght
            # step = header['CD1_1'] #step in wavelenght
            step = header['CDELT1']  # increment per pixel
            w0, dw, n = start_wave, step, len(flux)
            w = start_wave + step * n
            wave = np.linspace(w0, w, n, endpoint=False)
        # These types are produced by MOOGme (fits format).
        elif fname[-5:] == '.spec':
            hdulist = fits.open(fname)
            x = hdulist[1].data
            flux = x['flux']
            flux = np.array(flux)
            flux = flux.astype(np.float)  # flux data in the primary
            wave = x['wavelength']
            wave = np.array(wave)
            wave = wave.astype(np.float)
            #flux, wave = pyasl.dopplerShift(w, f, -0.15, edgeHandling="firstlast")

        # Cut observations to the intervals of the synthesis
        delta_l = wave[1] - wave[0]
        wavelength_obs = wave[np.where((wave >= float(start_synth)) & (wave <= float(end_synth)))]
        flux_obs = flux[np.where((wave >= float(start_synth)) & (wave <= float(end_synth)))]
        return wavelength_obs, flux_obs, delta_l

    else:
        print('Spectrum is not in acceptable format. Convert to ascii of fits.')
        wavelength_obs, flux_obs, delta_l = (None, None, None)
    return wavelength_obs, flux_obs, delta_l


def read_obs_intervals(obs_fname, r, snr=100, method='linear'):
    """Read only the spectral chunks from the observed spectrum
    This function does the same as read_observations but for the whole linelist.
    Input
    -----
    fname : filename of the spectrum. Currently only fits and text files accepted.
    r : ranges of wavelength intervals where the observed spectrum is cut
    (starting and ending wavelength)

    Output
    -----
    wavelength_obs : observed wavelength
    flux_obs : observed flux
    """
    from itertools import izip, tee

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    spec = [local_norm(obs_fname, ri, snr, method) for ri in r]
    #spec = [read_observations(obs_fname, ri[0], ri[1]) for ri in r]
    x_obs = np.hstack(np.vstack(spec).T[0])
    y_obs = np.hstack(np.vstack(spec).T[1])
    delta_l = x_obs[1] - x_obs[0]
    delta_l = round(delta_l, 3)
    #delta_l = 0.2

    if any(i == 0 for i in y_obs):
        print('Warning: Flux contains 0 values.')

    #y_obs[y_obs == 0] = 1
    snr = snr_apogee(obs_fname)
    print('SNR: %s' % int(snr))
    return x_obs, y_obs, delta_l


def plot(x_obs, y_obs, x, y, res=False, linelist='moog_apogee_atom_cn_clean.lst', intervals='intervals.lst'):
    """Function to plot synthetic spectrum.
    Input
    -----
    x_obs : observed wavelength
    y_obs : observed flux
    x : synthetic wavelength
    y : synthetic flux

    Output
    ------
    plots
    """

    # if nothing exists, pass
    if (x_obs is None) and (x is None):
        pass
    # if there is not observed spectrum, plot only synthetic (case 1, 3)
    if x_obs is None:
        plt.plot(x, y, label='synthetic')
        if res:
            sl = InterpolatedUnivariateSpline(x, y, k=1)
            ymodel = sl(x_obs)
            plt.plot(x_obs, (y_obs-ymodel)*10, label='residuals')
        plt.legend()
        plt.show()
    # if both exist
    else:
        #import lineid_plot
        #from synthetic import read_linelist

        #r, a = read_linelist(linelist, intervals)
        #line_wave  = a[:, 0]
        #line_label = a[:, 1]
        #line_label = np.array(map(str, line_label))
        #line_wave  = np.array(line_wave, dtype='float64')
        #sl = InterpolatedUnivariateSpline(x, y, k=1)
        #ymodel = sl(x_obs)
        #lineid_plot.plot_line_ids(x, y, line_wave, line_label, max_iter=1000)
        plt.plot(x, y, label='synthetic')
        plt.plot(x_obs, y_obs, marker='o', linestyle='-', markersize=1, label='observed')
        plt.grid(True)
        plt.xlabel(r'Wavelength $\AA{}$')
        plt.ylabel('Normalized flux')
        if res:
            sl = InterpolatedUnivariateSpline(x, y, k=1)
            ymodel = sl(x_obs)
            plt.plot(x_obs, (y_obs-ymodel)*10, label='residuals')

        #plt.legend()
        plt.show()
    return


def snr_apogee(fname):
    hdulist = fits.open(fname)
    header = hdulist[1].header
    try:
        snr = header['SNR']
    except KeyError as e:
        snr = 100
    return snr


def snr_pyastronomy(fname, plot=False):
    """Calculate SNR using intervals depending on giraffe mode.
    Input
    ----
    fname : spectrum
    plot : plot snr fit
    Output
    -----
    snr : snr value averaged from the continuum intervals
    """
    from PyAstronomy import pyasl

    def sub_snr(interval):
        '''Measure the SNR on a small interval
        Input
        -----
        interval : list
          Upper and lower limit on wavelength
        Output
        ------
        SNR : float
          The SNR in the small interval
        '''
        w1, w2 = interval
        wave_cut, flux_cut, l = read_observations(fname, w1, w2)
        num_points = int(len(flux_cut)/4)
        if num_points != 0:
            snrEsti1 = pyasl.estimateSNR(wave_cut, flux_cut, num_points, deg=2, controlPlot=plot)
            return snrEsti1["SNR-Estimate"]
        else:
            return 0

    intervals = [[5744, 5746], [6048, 6052], [6068, 6076], [6682, 6686], [6649, 6652],
                [6614, 6616], [5438.5, 5440], [5449.5, 5051], [5458, 5459.25],
                [5498.3, 5500],   [5541.5, 5542.5]]

    snr = []
    for interval in intervals:
        snr.append(sub_snr(interval))

    if not len(snr):
        snr = None
    else:
        snr = [value for value in snr if value != 0]
        snr_clean = [value for value in snr if not np.isnan(value)]
        snr_total = np.average(snr_clean)
        snr = round(snr_total, 1)
    return snr
