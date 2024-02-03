import copy
import dask
import itertools
import json
import numpy as np
import os
import sys
import time
import warnings

from glob import glob
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import (
                                  LinearRegression,
                                  RANSACRegressor
                                  )
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def _get_ref_peaks():
    # Peaks for the HG-2 calibration light source
    # Taken from the HG-2 manual
    hg2_peaks = np.array([
                          253.652, 296.728, 302.150, 313.155, 334.148, 365.015,
                          404.656, 407.783, 435.833, 546.074, 576.960, 579.066,
                          696.543, 706.722, 714.704, 727.294, 738.398, 750.387,
                          763.511, 772.376, 794.818, 800.616, 811.531, 826.452,
                          842.465, 852.144, 866.794, 912.297, 922.450
                          ])

    return hg2_peaks


def _fit_linear(peaks_px, position, grating=600, detector='newton971'):
    # Spectral dispersion based on grating used (nm/mm)
    if grating == 300:
        spec_disp = 10.12
    elif grating == 600:
        spec_disp = 4.94
    else:
        error('Only 300 gr/mm and 600 gr/mm configured for now!')

    if detector == 'newton971':
        # Detector pixel size (mm/px)
        det_size = 16E-3

        # Number of columns (px)
        det_cols = 1600
    else:
        error('Only Andor Newton EMCCD 971 configured for now!')

    # Spectral resolution (nm/px)
    spec_res = spec_disp * det_size

    # Approximate the X-axis wavelength values (nm)
    xaxis = (spec_res * np.arange(-det_cols/2, det_cols/2)) + position

    # Approximate peak positions (nm)
    peaks_nm = xaxis[peaks_px]

    return peaks_nm


def _remove_outlier_peaks(peaks, positions, winSize, threshold):
    # Number of datasets
    nsets = len(positions)
    train_inds = np.arange(nsets).tolist()

    # Initialize peaks list (w/o redundancies)
    use_peaks = [None] * len(peaks)

    # Get reference peak values
    hg2_peaks = _get_ref_peaks()

    # Remove redundancies
    for j in train_inds:
        # Crop down HG2 peak list (using 270 nm spectral dispersion over CCD)
        ind = np.logical_and(
                             hg2_peaks > (positions[j] - winSize/2),
                             hg2_peaks < (positions[j] + winSize/2)
                             )
        ref_peaks = hg2_peaks[ind]

        # Get first order approximation
        peaks_nm_lin = _fit_linear(peaks[j], positions[j])

        # Closest reference peaks
        ref_closest = np.array([
                                ref_peaks[np.argmin(abs(x-ref_peaks))]
                                for x in peaks_nm_lin
                                ])

        # Initialize quadratic model for RANSAC
        ransac_model = make_pipeline(
                                     PolynomialFeatures(degree=2),
                                     LinearRegression()
                                     )
        # Minimum number of samples
        min_samples = np.unique(ref_closest, return_counts=True)[1].max()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            ransac = RANSACRegressor(
                                     ransac_model,
                                     min_samples=min_samples,
                                     residual_threshold=threshold,
                                     stop_probability=0.999
                                     )
            ransac.fit(peaks[j].reshape(-1,1), ref_closest);

        use_peaks[j] = peaks[j][ransac.inlier_mask_]

    return use_peaks


def _train_err(peaks, refs):
    z, xval = _train_data(peaks, refs)
    err = _rmse(refs, xval[peaks])

    return err, z


def _train_data(peaks, refs):
    z = np.polyfit(peaks, refs, 2)
    xval = np.arange(0, 1600, 1)
    xval_nm = np.polyval(z, xval)

    return z, xval_nm


def _rmse(true, pred):
    return np.sqrt(mse(true, pred))


def _find_set(peaks, refs):
    tmp = copy.deepcopy(refs.tolist())
    closest_peaks = []

    for x in peaks:
        ind = np.argmin(abs(np.array(tmp) - x))
        closest_peaks.append(tmp[ind])
        tmp.pop(ind)

    return closest_peaks


def _process(fp, lines, positions, minPks, prom, thresh, grating, detector):
    # Get peak pixel positions from each data set
    peaks, _ = zip(*[
                     find_peaks(x, distance=10, prominence=prom)
                     for x in lines
                     ])

    # Window to use for peak cropping
    winSize = 300

    # Crop down the peak list to remove any outliers
    use_peaks = _remove_outlier_peaks(peaks, positions, winSize, thresh[0])

    # Number of peaks found
    npeaks = [len(x) for x in use_peaks]

    # Initialize lists
    nsets = len(positions)
    train_inds = np.arange(nsets).tolist()
    train_ref_best = [None] * nsets
    poly_best = [None] * nsets
    xval_nm = [None] * nsets
    train_rmse = [None] * nsets

    # Indices of training data sets (used for removing bad fits)
    keep_set = np.arange(0, nsets, 1).tolist()

    # Get reference peak values
    hg2_peaks = _get_ref_peaks()

    for j in train_inds:
        # Make sure there are enough peaks
        if npeaks[j] < minPks:
            print(f'{fp[j].split("/")[-1]} thrown out (<{minPks} peaks)')
            keep_set.remove(j)
            continue

        # Initialize lists
        train_peaks = use_peaks[j]
        poly = []
        train_ref = []

        # Crop down HG2 peak list (using 270 nm spectral dispersion over CCD)
        ind = np.logical_and(
                             hg2_peaks > (positions[j] - winSize/2),
                             hg2_peaks < (positions[j] + winSize/2)
                             )
        ref_peaks = hg2_peaks[ind]

        # Run all combinations for the training set
        start = time.time()
        for combo in itertools.combinations(ref_peaks, npeaks[j]):
            err, z = _train_err(use_peaks[j], combo)

            if err < thresh[1]:
                poly.append(z)
                train_ref.append(combo)

        # Indices to use for testing
        test_inds = list(set(np.arange(nsets)) - set([j]))
        test_err = [None] * (nsets-1)

        # Test out the trained fits on the other datasets
        for i, test_ind in enumerate(test_inds):
            # Get offset for the different grating position
            offset = positions[test_ind] - positions[j]

            # Crop down HG2 peak list
            ind = np.logical_and(
                                 hg2_peaks > (positions[test_ind] - winSize/2),
                                 hg2_peaks < (positions[test_ind] + winSize/2)
                                 )
            ref_peaks = hg2_peaks[ind]

            # Calculate error for each of the best trained models
            test_err[i] = []
            for p in poly:
                test_peaks = np.polyval(p, use_peaks[test_ind]) + offset
                test_peaks_closest = _find_set(test_peaks, ref_peaks)
                test_err[i].append(_rmse(test_peaks_closest, test_peaks))

        # Only use the training models that work for all testing sets
        test_err = np.array(test_err)

        # Find best model if it exists
        try:
            best_ind = np.sum(test_err, axis=0).argmin()
        except:
            print(f'{fp[j].split("/")[-1]} thrown out (could not fit)')
            keep_set.remove(j)
            continue

        # Get best training reference peaks and fits
        train_ref_best[j] = train_ref[best_ind]
        poly_best[j] = poly[best_ind]
        xval_nm[j] = np.polyval(poly_best[j], np.arange(1600))

        # Calculate final training RMSE
        train_rmse[j] = _rmse(
                              train_ref_best[j],
                              xval_nm[j][use_peaks[j]]
                              )
        elapsed = time.time() - start
        print(f'{fp[j].split("/")[-1]} finished: {elapsed/60:0.2f} min')

    # Package output
    out = {
           'Files': np.array(fp)[keep_set].tolist(),
           'Position': np.array(positions)[keep_set].tolist(),
           'Peaks/px': [use_peaks[x].tolist() for x in keep_set],
           'Reference/nm': np.array(train_ref_best, dtype=object)
                           [keep_set].tolist(),
           'polyfit': [poly_best[x].tolist() for x in keep_set],
           'xval/nm': [xval_nm[x].tolist() for x in keep_set],
           'RMSE': np.array(train_rmse)[keep_set].tolist(),
           'Grating': grating,
           'Detector': detector
           }

    return out, np.array(lines)[keep_set]


def _plot_calib(calib, lines):
    # Number of datasets
    nsets = len(calib['Position'])

    # Get reference peak values
    hg2_peaks = _get_ref_peaks()

    # Colors
    cdata = 'black'
    cpeaks = 'hotpink'
    cfit = 'grey'
    corder1 = 'lightcoral'
    corder2 = 'limegreen'
    corder3 = 'cornflowerblue'

    fig, ax = plt.subplots(nsets, 1, figsize=(8, nsets*1.4))
    plt.subplots_adjust(
                        left=0.095, right=0.98,
                        bottom=0.1, top=0.945,
                        hspace=0.66
                        )

    for i, a in enumerate(np.ravel(ax)):
        # Visualize spectra
        a.plot(calib['xval/nm'][i], lines[i], color=cdata, alpha=1, zorder=99)

        # Visualize peaks
        [
         a.plot(
                np.array(calib['xval/nm'][i])[calib['Peaks/px'][i]],
                lines[i][calib['Peaks/px'][i]],
                color=cpeaks,
                linestyle='',
                marker='x',
                )
         for x in calib['Reference/nm'][i]
         ]

        # Get current xlim
        xlim = a.get_xlim()

        # Visualize fits
        [
         a.axvline(
                   x,
                   color=cfit,
                   alpha=1,
                   linestyle='--',
                   zorder=1
                   )
         for x in calib['Reference/nm'][i]
         ]

        # Visualize unused first/second/third order peaks
        for p in hg2_peaks:
            if p in calib['Reference/nm'][i]:
                continue

            if np.logical_and(1*p > xlim[0], 1*p < xlim[1]):
                a.axvline(1*p, color=corder1, ls='--', alpha=0.8, zorder=0)

            if np.logical_and(2*p > xlim[0], 2*p < xlim[1]):
                a.axvline(2*p, color=corder2, ls='-.', alpha=0.8, zorder=0)

            if np.logical_and(3*p > xlim[0], 3*p < xlim[1]):
                a.axvline(3*p, color=corder3, ls=':', alpha=0.8, zorder=0)

        position = calib['Position'][i]
        rmse = calib['RMSE'][i]
        a.set_title(f'Position = {position} nm: RMSE = {rmse:0.3f} nm')
        a.set_xlim(xlim)

    # Create legend using dummy items
    cross = mlines.Line2D(
                          [], [],
                          color=cpeaks,
                          marker='x',
                          linestyle='None',
                          label='Measured peaks'
                          )
    lineUsed = mlines.Line2D(
                             [], [],
                             color=cfit,
                             linestyle='--',
                             alpha=1,
                             label='Fitted peaks'
                             )
    line1 = mlines.Line2D(
                          [], [],
                          color=corder1,
                          linestyle='--',
                          alpha=1,
                          label='1$^{st}$ order (unused)'
                          )
    line2 = mlines.Line2D(
                          [], [],
                          color=corder2,
                          linestyle='-.',
                          alpha=1,
                          label='2$^{nd}$ order (unused)'
                          )
    line3 = mlines.Line2D(
                          [], [],
                          color=corder3,
                          linestyle=':',
                          alpha=1,
                          label='3$^{rd}$ order (unused)'
                          )
    np.ravel(ax)[0].legend(
                           handles=[cross, lineUsed, line1, line2, line3],
                           fontsize='x-small',
                           ncols=2
                           )

    # X axis label
    np.ravel(ax)[-1].set_xlabel('Wavelength (nm)')

    plt.show()


def _save(calib, save_fld, save_name):
    with open(f'{save_fld}/{save_name}.json', 'w') as handle:
        json.dump(calib, handle, indent=4)

    return


def autocalibrate(
                  fp,
                  lines,
                  positions,
                  minPeaks=5,
                  prominence=500,
                  thresholds=(0.1, 5),
                  save_fld=None,
                  save_name='xeol_calibration',
                  plot=False,
                  grating=600,
                  detector='newton971'
                  ):
    # Check thresholds input
    if len(thresholds) == 1:
        thresholds = np.repeat(thresholds, 2)

    # Run calibration
    calib, lines = _process(
                            fp,
                            lines,
                            positions,
                            minPeaks,
                            prominence,
                            thresholds,
                            grating,
                            detector
                            )

    # Save data (make sure output directory exists!)
    if save_fld is not None:
        if os.path.isdir(save_fld):
            _save(calib, save_fld, save_name)
        else:
            print(f'Create {save_fld} and then re-run this!')

    # Visualize calibration
    if plot:
        _plot_calib(calib, lines)

    return calib


def manual_calibration(
                       fp,
                       lines,
                       positions,
                       peaks_nm,
                       peaks_px,
                       save_fld=None,
                       save_name='xeol_calibration',
                       plot=False,
                       grating=600,
                       detector='newton971'
                       ):
    # Fit each of the set of peaks
    ndim = np.ndim(np.array(peaks_nm, dtype=object))

    if ndim == 1:
        err, z = _train_err(peaks_px, peaks_nm)

        # Convert to list (of lists) for consistency later on
        positions = [positions]
        peaks_nm = [peaks_nm]
        peaks_px = [peaks_px]
        err = [err]
        z = [z]
    else:
        err, z = zip(*[_train_err(x, y) for px, nm in zip(peaks_nm, peaks_px)])

    # X-axis wavelength values
    xval_nm = [np.polyval(coeff, np.arange(0, 1600, 1)) for coeff in z]

    # Package output
    calib = {
             'File': fp,
             'Position': [x.tolist() for x in positions],
             'Peaks/px': [x.tolist() for x in peaks_px],
             'Reference/nm': [x.tolist() for x in peaks_nm],
             'polyfit': [x.tolist() for x in z],
             'xval/nm': [x.tolist() for x in xval_nm],
             'RMSE': [x.tolist() for x in err],
             'Grating': grating,
             'Detector': detector
             }

    # Save data (make sure output directory exists!)
    if save_fld is not None:
        if os.path.isdir(save_fld):
            _save(calib, save_fld, save_name)
        else:
            print(f'Create {save_fld} and then re-run this!')

    # Visualize calibration
    if plot:
        _plot_calib(calib, lines)

    return calib


def apply_calibration(calib_fp, position):
    # Load in calibration data
    with open(calib_fp, 'r') as handle:
        calib = json.load(handle)

    # Get closest position
    ind = np.argmin(abs(np.array(calib['Position']) - position))

    # Get offset
    offset = position - calib['Position'][ind]

    # Calculate wavelengths
    wavelengths = np.polyval(calib['polyfit'][ind], np.arange(0, 1600, 1))

    # Apply offset
    wavelengths += offset

    return np.float32(wavelengths)
