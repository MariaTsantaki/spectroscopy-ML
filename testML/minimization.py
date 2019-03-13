#!/usr/bin/env python
# -*- coding: utf8 -*-

# My imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpfit import mpfit

class minimize_ML:
    def __init__(self, flux, model, p0=[5777, 4.44, 0.0, 0.0], scale=False):
        self.p0    = p0
        self.flux  = flux
        self.model = model

    def _convergence_info(self, res, parinfo):
        """
        Information on convergence. All values greater than zero can
        represent success (however status == 5 may indicate failure to
        converge).
        If the fit is unweighted (i.e. no errors were given, or the weights
        were uniformly set to unity), then .perror will probably not represent
        the true parameter uncertainties.
        *If* you can assume that the true reduced chi-squared value is unity --
        meaning that the fit is implicitly assumed to be of good quality --
        then the estimated parameter uncertainties can be computed by scaling
        .perror by the measured chi-squared value.
        """

        self.x_red = round((res.fnorm / self.dof), 4)
        #print('Iterations: %s' % res.niter)
        #print('Value of the summed squared residuals: %s' % res.fnorm)
        #print('Reduced chi squared: %s' % self.x_red)
        #print('Fitted parameters with uncertainties:')
        # scaled uncertainties
        pcerror = res.perror * np.sqrt(res.fnorm / self.dof)
        teff  = round(float(res.params[0]), 0)
        logg  = round(float(res.params[1]), 2)
        feh   = round(float(res.params[2]), 3)
        alpha = round(float(res.params[3]), 3)
        #scaled error
        erteff  = round(float(res.perror[0]), 0)
        erlogg  = round(float(res.perror[1]), 2)
        erfeh   = round(float(res.perror[2]), 3)
        eralpha = round(float(res.perror[3]), 3)

        # Save only the scaled error
        self.parameters = [teff, logg, feh, alpha]
        #print('    Teff:{:8.1f}   logg: {:1.2f}   [Fe/H]: {:1.2f}   [a/Fe]: {:1.2f}'.format(*self.parameters))
        return self.parameters

    def _myfunct(self, p, y_obs=None, fjac=None):
        '''Function that return the weighted deviates (to be minimized).

        Input
        ----
        p : list
          Parameters for the ML
        flux : ndarray
          Observed flux

        Output
        -----
        (flux-y)/err : ndarray
          Model deviation from observation
        '''
        y = self.model.get_spectrum(p)
        err = np.zeros(len(y)) + 0.001
        status = 0
        # Print parameters at each function call
        #print('    Teff:{:8.1f}   logg: {:1.2f}   [Fe/H]: {:1.2f}   [a/Fe]: {:1.2f}'.format(*p))
        chi2 = (y - y_obs)/err
        return([status, chi2])

    def minimize(self):
        # Set PARINFO structure for all 4 free parameters for mpfit
        teff_info  = {'parname':'Teff',   'limited': [1, 1], 'limits': [3950, 7000], 'step': 100,  'mpside': 2, 'mpprint': 0}
        logg_info  = {'parname':'logg',   'limited': [1, 1], 'limits': [1.2, 4.9],   'step': 0.10, 'mpside': 2, 'mpprint': 0}
        feh_info   = {'parname':'[M/H]',  'limited': [1, 1], 'limits': [-2.5, 0.7],  'step': 0.05, 'mpside': 2, 'mpprint': 0}
        alpha_info = {'parname':'[a/Fe]', 'limited': [1, 1], 'limits': [-0.3, 0.4],  'step': 0.05, 'mpside': 2, 'mpprint': 0}
        self.parinfo = [teff_info, logg_info, feh_info, alpha_info]
        fa = {'y_obs': self.flux}

        # Minimization starts here.
        self.m = mpfit(self._myfunct, xall=self.p0, parinfo=self.parinfo, ftol=1e-10, xtol=1e-8, gtol=1e-8, functkw=fa, maxiter=50, quiet=1)
        self.dof = len(self.flux) - len(self.m.params)
        self.params = self._convergence_info(self.m, self.parinfo)
        #print(self.params)
        #print(self._convergence_info(self.m, self.parinfo))
        return self.params
