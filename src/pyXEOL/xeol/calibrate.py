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
from sklearn.metrics import (
                             mean_squared_error as mse,
                             r2_score
                             )


def _train_data(peaks, refs):
    z = np.polyfit(peaks, refs, 2)
    xval = np.arange(0, 1600, 1)
    xval_nm = np.polyval(z, xval)

    return z, xval_nm


def _train_acc(peaks, refs):
    z, xval = _train_data(peaks, refs)
    acc = r2_score(refs, xval[peaks])

    return acc, z


def _find_set(peaks, refs):
    tmp = copy.deepcopy(refs.tolist())
    closest_peaks = []

    for x in peaks:
        ind = np.argmin(abs(np.array(tmp) - x))
        closest_peaks.append(tmp[ind])
        tmp.pop(ind)

    return closest_peaks


def _process(fp, lines, gratings, prom, r2):
    # Number of datasets
    nsets = len(gratings)

    # Get peak pixel positions from each data set
    peaks, _ = zip(*[
                     find_peaks(x, distance=20, prominence=prom)
                     for x in lines
                     ])
    npeaks = [len(x) for x in peaks]

    hg2_peaks = np.array([
                          253.652, 296.728, 302.150, 313.155, 334.148, 365.015,
                          404.656, 407.783, 435.833, 546.074, 576.960, 579.066,
                          696.543, 706.722, 714.704, 727.294, 738.398, 750.387,
                          763.511, 772.376, 794.818, 800.616, 811.531, 826.452,
                          842.465, 852.144, 866.794, 912.297, 922.450
                          ])
    train_inds = np.arange(nsets).tolist()

    train_ref_best = [None] * nsets
    poly_best = [None] * nsets
    xval_nm = [None] * nsets
    train_rmse = [None] * nsets

    # Window to use for peak cropping
    winSize = 300

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
            acc, z = _train_acc(peaks[train_ind], combo)

            if acc > r2:
                poly.append(z)
                train_ref.append(combo)

        # Indices to use for testing
        test_inds = list(set(np.arange(nsets)) - set([train_ind]))
        test_r2 = [None] * (nsets-1)

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

            # Calculate R^2 for each of the best trained models
            test_r2[i] = []
            for p in poly:
                test_peaks = np.polyval(p, peaks[test_ind]) + offset
                test_peaks_closest = _find_set(test_peaks, ref_peaks)
                test_r2[i].append(r2_score(test_peaks_closest, test_peaks))

        elapsed = time.time() - start
        print(f'{fp[j].split("/")[-1]} finished: {elapsed/60:0.2f} min')

        # Only use the training models that work for all testing sets
        test_r2 = np.array(test_r2)
        good_r2 = np.prod(test_r2 > r2, axis=0)
        good_r2_ind = np.nonzero(good_r2)[0]

        # Index of best training model
        best_ind = good_r2_ind[test_r2[:, good_r2_ind].mean(axis=0).argmax()]

        # Get best training reference peaks and fits
        train_ref_best[j] = train_ref[best_ind]
        poly_best[j] = poly[best_ind]
        xval_nm[j] = np.polyval(poly_best[j], np.arange(1600))

        # Calculate final training RMSE
        train_mse = mse(train_ref_best[j], xval_nm[j][peaks[train_ind]])
        train_rmse[j] = np.sqrt(train_mse)

    # Package output
    out = {
           'Files': fp,
           'Grating': gratings,
           'Peaks/px': [x.tolist() for x in peaks],
           'Reference/nm': train_ref_best,
           'polyfit': [x.tolist() for x in poly_best],
           'xval/nm': [x.tolist() for x in xval_nm],
           'RMSE': train_rmse
           }

    return out


def _plot_calib(calib, lines):
    # Number of datasets
    nsets = len(calib['Grating'])

    fig, ax = plt.subplots(nsets, 1, figsize=(8, nsets*1.4))
    plt.subplots_adjust(
                        left=0.095, right=0.98,
                        bottom=0.1, top=0.945,
                        hspace=0.66
                        )

    for i, a in enumerate(ax):
        # Visualize spectra
        a.plot(calib['xval/nm'][i], lines[i], color='k', alpha=0.6)

        # Visualize peaks
        [
         a.plot(
                np.array(calib['xval/nm'][i])[calib['Peaks/px'][i]],
                lines[i][calib['Peaks/px'][i]],
                color='b',
                linestyle='',
                marker='x',
                )
         for x in calib['Reference/nm'][i]
         ]

        # Visualize fits
        [
         a.axvline(
                   x,
                   color='r',
                   alpha=0.5,
                   linestyle='--',
                   )
         for x in calib['Reference/nm'][i]
         ]

        grating = calib['Grating'][i]
        rmse = calib['RMSE'][i]
        a.set_title(f'Grating = {grating} nm: RMSE = {rmse:0.3f} nm')

    # Create legend using dummy items
    blue_cross = mlines.Line2D(
                               [], [],
                               color='b',
                               marker='x',
                               linestyle='None',
                               label='Found peaks'
                               )
    red_line = mlines.Line2D(
                             [], [],
                             color='r',
                             linestyle='--',
                             alpha=0.5,
                             label='Fitted peaks'
                             )
    ax[0].legend(handles=[blue_cross, red_line])

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
                  r2=0.9,
                  save_fld=None,
                  plot=False
                  ):
    # Run calibration
    calib = _process(fp, lines, gratings, prom, r2)

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
