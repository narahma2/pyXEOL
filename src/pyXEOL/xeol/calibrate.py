import copy
import dask
import itertools
import json
import numpy as np
import os
import sys
import time

from glob import glob
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error as mse


# Peaks for the HG-2 calibration light source
hg2_peaks = np.array([
                      253.652, 296.728, 302.150, 313.155, 334.148, 365.015,
                      404.656, 407.783, 435.833, 546.074, 576.960, 579.066,
                      696.543, 706.722, 714.704, 727.294, 738.398, 750.387,
                      763.511, 772.376, 794.818, 800.616, 811.531, 826.452,
                      842.465, 852.144, 866.794, 912.297, 922.450
                      ])


def _rmse(true, pred):
    return np.sqrt(mse(true, pred))


def _train_data(peaks, refs):
    z = np.polyfit(peaks, refs, 2)
    xval = np.arange(0, 1600, 1)
    xval_nm = np.polyval(z, xval)

    return z, xval_nm


def _train_err(peaks, refs):
    z, xval = _train_data(peaks, refs)
    err = _rmse(refs, xval[peaks])

    return err, z


def _find_set(peaks, refs):
    tmp = copy.deepcopy(refs.tolist())
    closest_peaks = []

    for x in peaks:
        ind = np.argmin(abs(np.array(tmp) - x))
        closest_peaks.append(tmp[ind])
        tmp.pop(ind)

    return closest_peaks


def _process(fp, lines, gratings, prom, threshold):
    # Number of datasets
    nsets = len(gratings)

    # Get peak pixel positions from each data set
    peaks, _ = zip(*[
                     find_peaks(x, distance=20, prominence=prom)
                     for x in lines
                     ])
    npeaks = [len(x) for x in peaks]

    train_inds = np.arange(nsets).tolist()

    train_ref_best = [None] * nsets
    poly_best = [None] * nsets
    xval_nm = [None] * nsets
    train_rmse = [None] * nsets

    # Window to use for peak cropping
    winSize = 300

    # Indices of training data sets (used for cropping bad fits later)
    use_inds = np.arange(0, nsets, 1).tolist()

    for j, train_ind in enumerate(train_inds):
        train_peaks = peaks[train_ind]
        poly = []
        train_ref = []

        # Crop down HG2 peak list (using 270 nm spectral dispersion over CCD)
        ind = np.logical_and(
                             hg2_peaks > (gratings[train_ind] - winSize/2),
                             hg2_peaks < (gratings[train_ind] + winSize/2)
                             )
        ref_peaks = hg2_peaks[ind]

        # Run all combinations for the training set
        start = time.time()
        for combo in itertools.combinations(ref_peaks, npeaks[train_ind]):
            err, z = _train_err(peaks[train_ind], combo)

            if err < threshold:
                poly.append(z)
                train_ref.append(combo)

        # Indices to use for testing
        test_inds = list(set(np.arange(nsets)) - set([train_ind]))
        test_err = [None] * (nsets-1)

        # Test out the trained fits on the other datasets
        for i, test_ind in enumerate(test_inds):
            # Get offset for the different grating position
            offset = gratings[test_ind] - gratings[train_ind]

            # Crop down HG2 peak list
            ind = np.logical_and(
                                 hg2_peaks > (gratings[test_ind] - winSize/2),
                                 hg2_peaks < (gratings[test_ind] + winSize/2)
                                 )
            ref_peaks = hg2_peaks[ind]

            # Calculate error for each of the best trained models
            test_err[i] = []
            for p in poly:
                test_peaks = np.polyval(p, peaks[test_ind]) + offset
                test_peaks_closest = _find_set(test_peaks, ref_peaks)
                test_err[i].append(_rmse(test_peaks_closest, test_peaks))

        # Only use the training models that work for all testing sets
        test_err = np.array(test_err)
        good_err = np.prod(test_err < threshold, axis=0)
        good_ind = np.nonzero(good_err)[0]

        # Populate only if fit is possible
        if len(good_ind) > 0:
            # Index of best training model
            best_ind = good_ind[test_err[:, good_ind].mean(axis=0).argmin()]

            # Get best training reference peaks and fits
            train_ref_best[j] = train_ref[best_ind]
            poly_best[j] = poly[best_ind]
            xval_nm[j] = np.polyval(poly_best[j], np.arange(1600))

            # Calculate final training RMSE
            train_rmse[j] = _rmse(
                                  train_ref_best[j],
                                  xval_nm[j][peaks[train_ind]]
                                  )

            elapsed = time.time() - start
            print(f'{fp[j].split("/")[-1]} finished: {elapsed/60:0.2f} min')
        else:
            print(f'{fp[j].split("/")[-1]} thrown out (could not fit)')
            train_ref_best[j] = None
            poly_best[j] = None
            xval_nm[j] = None
            train_rmse[j] = None
            use_inds.remove(j)

    # Package output
    out = {
           'Files': np.array(fp)[use_inds].tolist(),
           'Grating': np.array(gratings)[use_inds].tolist(),
           'Peaks/px': [x.tolist() for x in peaks if x is not None],
           'Reference/nm': np.array(train_ref_best, dtype=object)
                           [use_inds].tolist(),
           'polyfit': [x.tolist() for x in poly_best if x is not None],
           'xval/nm': [x.tolist() for x in xval_nm if x is not None],
           'RMSE': np.array(train_rmse)[use_inds].tolist()
           }

    return out


def _plot_calib(calib, lines):
    # Number of datasets
    nsets = len(calib['Grating'])

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

    for i, a in enumerate(ax):
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
                   color=fit,
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

        grating = calib['Grating'][i]
        rmse = calib['RMSE'][i]
        a.set_title(f'Grating = {grating} nm: RMSE = {rmse:0.3f} nm')
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
    ax[0].legend(
                 handles=[cross, lineUsed, line1, line2, line3],
                 fontsize='x-small',
                 ncols=2
                 )

    # X axis label
    ax[-1].set_xlabel('Wavelength (nm)')

    plt.show()


def _save(calib, save_fld):
    with open(f'{save_fld}/xeol_calibration.json', 'w') as handle:
        json.dump(calib, handle, indent=4)

    return


def autocalibrate(
                  fp,
                  lines,
                  gratings,
                  prom=500,
                  threshold=3,
                  save_fld=None,
                  plot=False
                  ):
    # Run calibration
    calib = _process(fp, lines, gratings, prom, threshold)

    # Save data (make sure output directory exists!)
    if save_fld is not None:
        if os.path.isdir(save_fld):
            _save(calib, save_fld)
        else:
            print(f'Create {save_fld} and then re-run this!')

    # Visualize calibration
    if plot:
        _plot_calib(calib, lines)

    return calib


def apply_calibration(calib_fp, grating):
    # Load in calibration data
    with open(calib_fp, 'r') as handle:
        calib = json.load(handle)

    # Get closest grating
    ind = np.argmin(abs(np.array(calib['Grating']) - grating))

    # Get offset
    offset = grating - calib['Grating'][ind]

    # Calculate wavelengths
    wavelengths = np.polyval(calib['polyfit'][ind], np.arange(0, 1600, 1))

    # Apply offset
    wavelengths += offset

    return np.float32(wavelengths)
