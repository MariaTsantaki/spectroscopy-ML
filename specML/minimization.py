#!/usr/bin/env python
# -*- coding: utf8 -*-

# My imports
from __future__ import division
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpfit import mpfit
from sklearn import preprocessing

def minimize_ML(clf, y_obs, scale=False):
    '''Minimize a synthetic spectrum to an observed

     Input
     -----
     clf : ndarray
       Model to fit
     y_obs : ndarray
       Observed flux

     Output
     -----
     params : list
       Final parameters
    '''


    def convergence_info(res, parinfo, dof):
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

        if res.status == -16:
            print('status = %s : A parameter or function value has become infinite or an undefined number.' % res.status)
        if -15 <= res.status <= -1:
            print('status = %s : MYFUNCT or iterfunct functions return to terminate the fitting process. ' % res.status)
        if res.status == 0:
            print('status = %s : Improper input parameters.' % res.status)
        if res.status == 1:
            print('status = %s : Both actual and predicted relative reductions in the sum of squares are at most ftol.' % res.status)
        if res.status == 2:
            print('status = %s : Relative error between two consecutive iterates is at most xtol.' % res.status)
        if res.status == 3:
            print('status = %s : Conditions for status = 1 and status = 2 both hold.' % res.status)
        if res.status == 4:
            print('status = %s : The cosine of the angle between fvec and any column of the jacobian is at most gtol in absolute value.' % res.status)
        if res.status == 5:
            print('status = %s : The maximum number of iterations has been reached.' % res.status)
        if res.status == 6:
            print('status = %s : ftol is too small.' % res.status)
        if res.status == 7:
            print('status = %s : xtol is too small.' % res.status)
        if res.status == 8:
            print('status = %s : gtol is too small.' % res.status)

        x_red = round((res.fnorm / dof), 4)
        print('Iterations: %s' % res.niter)
        #print('Value of the summed squared residuals: %s' % res.fnorm)
        print('Reduced chi squared: %s' % x_red)
        #print('Fitted parameters with uncertainties:')
        # scaled uncertainties
        pcerror = res.perror * np.sqrt(res.fnorm / dof)
        teff  = round(float(res.params[0]), 0)
        logg  = round(float(res.params[1]), 2)
        feh   = round(float(res.params[2]), 3)
        alpha = round(float(res.params[3]), 3)
        #vsini = round(float(res.params[5]), 1)
        #scaled error
        erteff  = round(float(res.perror[0]), 0)
        erlogg  = round(float(res.perror[1]), 2)
        erfeh   = round(float(res.perror[2]), 3)
        eralpha = round(float(res.perror[3]), 3)
        #ervsini = round(float(res.perror[5]), 1)
        # Save only the scaled error
        parameters = [teff, logg, feh, alpha]
        for i, x in enumerate(res.params):
                    print( "\t%s: %s +- %s (scaled error +- %s)" % (parinfo[i]['parname'], round(x, 3), round(res.perror[i], 3), round(pcerror[i], 3)))
        return parameters


    def myfunct(p, y_obs=y_obs, y_obserr=0.1, clf=clf, scale=scale, fjac=None):
        '''Function that return the weighted deviates (to be minimized).

        Input
        ----
        p : list
          Parameters for the ML
        y_obs : ndarray
          Observed flux

        Output
        -----
        (y_obs-y)/err : ndarray
          Model deviation from observation
        '''

        p = np.append(p, (p[0]**2, p[1]**2, p[2]**2, p[0]*p[1], p[0]*p[2], p[1]*p[2]))
        if scale is True:
            scaler = preprocessing.StandardScaler().fit(np.array(p).reshape(-1, 1))
            p = scaler.transform(np.array(p).reshape(-1, 1))
            y = clf.predict(np.array(p).reshape(1, -1))[0]
        else:
            y = clf.predict(np.array(p).reshape(1, -1))[0]
        plt.plot(y, label='model')
        plt.plot(y_obs, label='input')
        plt.legend()
        plt.show()
        # Error on the flux #needs corrections
        err = np.zeros(len(y_obs)) + y_obserr
        status = 0
        #Print parameters at each function call
        #print('    Teff:{:8.1f}   logg: {:1.2f}   [Fe/H]: {:1.2f}   vsini: {:1.2f}'.format(*p))
        return([status, (y-y_obs)/err])


    # Set PARINFO structure for all 4 free parameters for mpfit
    # Teff, logg, feh, vsini
    teff_info  = {'parname':'Teff',   'limited': [1, 1], 'limits': [3000, 7000], 'step': 100,  'mpside': 2, 'mpprint': 0}
    logg_info  = {'parname':'logg',   'limited': [1, 1], 'limits': [3.0, 5.0],   'step': 0.1,  'mpside': 2, 'mpprint': 0}
    feh_info   = {'parname':'[M/H]',  'limited': [1, 1], 'limits': [-2.5, 0.6],  'step': 0.05, 'mpside': 2, 'mpprint': 0}
    alpha_info = {'parname':'[a/Fe]', 'limited': [1, 1], 'limits': [-0.5, 0.5],  'step': 0.05, 'mpside': 2, 'mpprint': 0}
    #vt_info    = {'parname':'vmic',  'limited': [1, 1], 'limits': [0.0, 9.9],   'step': 0.5,  'mpside': 2}
    #vmac_info  = {'parname':'vmac',  'limited': [1, 1], 'limits': [0.0, 20.0],  'step': 0.5,  'mpside': 2}
    #vsini_info = {'parname':'vsini', 'limited': [1, 1], 'limits': [0.0, 90.0],  'step': 1.0,  'mpside': 2}
    parinfo = [teff_info, logg_info, feh_info, alpha_info]

    yerr = 1 #arbitary value
    fa = {'y_obs': y_obs, 'y_obserr': yerr, 'clf': clf, 'scale': scale}
    p0 = [5777, 4.44, 0.0, 0.0]

    # Minimization starts here.
    # Measure time
    start_time = time.time()
    m = mpfit(myfunct, xall=p0, parinfo=parinfo, ftol=1e-5, xtol=1e-5, gtol=1e-15, functkw=fa, maxiter=20, quiet=1)
    dof = len(y_obs) - len(m.params)
    parameters = convergence_info(m, parinfo, dof)

    #plt.plot(y_obs, '-o', label='obs')
    #p = np.append(p, (p[0]**2, p[1]**2, p[2]**2, p[0]*p[1], p[0]*p[2], p[1]*p[2]))
    #y = clf.predict(np.array(p0).reshape(1, -1))[0]
    #plt.plot(y, '-o', label='sun')
    #y = clf.predict(np.array(parameters).reshape(1, -1))[0]
    #plt.plot(y, '-o', label='result')
    #plt.legend()
    #plt.show()
    return parameters
