import dask
import numpy as np
import os
import sys
import warnings
import xarray as xr
import zarr

from dask import array as da
from dask.diagnostics import ProgressBar
from dask_image.imread import imread
from glob import glob
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy.optimize import curve_fit
from shutil import rmtree
from sklearn.metrics import r2_score

from pyXEOL.specfun import (
                            baseline_arPLS,
                            df,
                            gauss,
                            gauss_p0,
                            gauss_p0_stack,
                            sigma2fwhm,
                            )
from pyXEOL.misc import create_folder


def process_stack_dask(in_fld, wavelengths, out_fld, fit=True, wl_crop=200):
    # Load in files and sort (just in case)
    fn = glob(f'{in_fld}/*')

    if '.tif' not in fn[0]:
        im = None
        ind = np.argsort([int(x.split('_')[-1]) for x in fn])
        fn = np.array(fn)[ind].tolist()
    else:
        im = f'{in_fld}/*.tif'
        fn = glob(f'{in_fld}/*.tif')

    # Continue if no files were loaded
    if not bool(fn):
        return 0

    # Delete output if it exists
    zfp = f'{out_fld}'
    if os.path.isdir(zfp):
        rmtree(zfp)

    # Process results
    print(f'{in_fld.split("/")[-1]} processing...')
    if fit:
        print('[0/7] Initiating...')
        _pipeline_image2spec(fn, im, zfp, wavelengths, 7, wl_crop)
        _pipeline_specfit(zfp)
    else:
        print('[0/3] Initiating...')
        _pipeline_image2spec(fn, im, zfp, wavelengths, 3, wl_crop)
        _clear_stdout(2)

    _update_stdout(f'{in_fld.split("/")[-1]} finished!')

    return 1


def process_single(fp, wavelengths, bg_corr=False, wl_crop=None, plot=True):
    if '.tif' in fp:
        raw2D = np.array(Image.open(fp)).astype(np.float32)
    else:
        raw2D = xr.open_dataset(fp)['array_data'].data.astype(np.float32)
        raw2D = raw2D.squeeze()

    # Crop down images
    if wl_crop is not None:
        raw2D = raw2D[:,wl_crop:-wl_crop]

    # Baseline (scalar subtraction)
    bl = np.median(raw2D[:100,:100])
    raw1D = (raw2D - bl).sum(axis=0)

    # Background (vector) subtraction
    if bg_corr:
        bg = baseline_arPLS(raw1D, lam=10E8, ratio=0.1, niter=1E3)
        ds = raw1D - bg
    else:
        ds = raw1D

    # Initial guesses
    x0, y0, sigma = gauss_p0(ds, wavelengths, xcrop=50)
    p0 = [x0, y0, sigma]

    # Fit the data, output guesses if it doesn't work
    try:
        p0, _ = curve_fit(gauss, wavelengths, ds, p0=p0, jac=df)

        # Extract parameters
        x0, y0, sigma = p0
        status = 1
    except:
        status = 0
        pass

    # Plot results (if requested)
    if plot:
        vmin = np.percentile(raw2D, 1)
        vmax=  np.percentile(raw2D, 99)

        fig, ax = plt.subplots(2, 1)
        im = ax[0].imshow(
                          raw2D,
                          vmin=vmin, vmax=vmax,
                          cmap='plasma'
                          )
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)
        ax[0].set_title(f'{fp.split("/")[-1].split(".")[0]}')

        ax[1].plot(
                   wavelengths,
                   ds,
                   linestyle='-',
                   color='k',
                   alpha=0.8,
                   label='Data',
                   )
        if status:
            ax[1].plot(
                       wavelengths,
                       gauss(wavelengths, *p0),
                       linestyle='--',
                       color='r',
                       alpha=0.8,
                       label='Fit',
                       )
        ax[1].legend()

        ax[1].axvline(x0, linestyle='--', color='gray', alpha=0.8)
        ax[1].set_xlabel('Wavelength (nm)')
        ax[1].set_ylabel('Intensity (a.u.)')

        string = (
                  f'Center = {x0:0.2f} nm\n'
                  f'Peak = {y0:0.2f}\n'
                  f'FWHM = {sigma2fwhm(sigma):0.2f} nm'
                  )
        ax[1].text(0.05, 0.6, string, transform=ax[1].transAxes)

        plt.show()

    return (status, x0, y0, sigma)


def _pipeline_image2spec(fn, im, zfp, wavelengths, steps, wl_crop):
    # Load images or netCDF4 files
    _update_stdout(f'[1/{steps}] Loading images...')
    if im is not None:
        raw = imread(im)
    else:
        raw = xr.open_mfdataset(
                                fn, combine='nested',
                                concat_dim='numArrays',
                                engine='scipy',
                                parallel=True,
                                )['array_data'].data.astype(np.float32)

    # Crop down images
    raw = raw[:,:,wl_crop:-wl_crop]

    # Baseline (scalar subtraction)
    bl = da.median(raw[:,:100,:100], axis=(1,2))
    raw = (raw - bl[:, np.newaxis, np.newaxis]).sum(axis=1)

    # Background (vector) subtraction
    _update_stdout(f'[2/{steps}] Background subtraction...')
    bg = [da.from_delayed(
                          dask.delayed(baseline_arPLS)
                          (x, lam=10E8, ratio=0.1, niter=1E3),
                          shape=(raw.shape[1],),
                          dtype=raw.dtype
                          )
          for x in raw[:1000]
          ]
    bg = da.array(bg).mean(axis=0)
    ds = raw - bg

    # Save output
    _update_stdout(f'[3/{steps}] Saving processed images...')
    names = [x.split('/')[-1].split('.')[0] for x in fn]
    nums = [int(x.split('_')[-1]) for x in names]
    data_vars = {
                 'folder': fn[0].split('/')[-2],
                 'files': (['t'], names),
                 }
    coords = dict(t=nums)
    out1 = xr.Dataset(data_vars=data_vars, coords=coords)
    out1.to_zarr(zfp, mode='w', group='/')

    data_vars = {
                 'bg': (['x'], bg),
                 'data': (['t', 'x'], ds),
                 }
    coords = dict(x=wavelengths, t=nums)
    out2 = xr.Dataset(data_vars=data_vars, coords=coords)

    with ProgressBar():
        out2.to_zarr(zfp, mode='w', group='xeol')

    return


def _pipeline_specfit(zfp):
    _clear_stdout(1)

    # Load in the processed spectra data
    _update_stdout('[4/7] Initiating fitting...')
    xeol = xr.open_zarr(zfp, group='xeol')
    xeol_data = xeol['data']

    # Gaussian parameter estimates and bounds using a subset of the XEOL data
    xeol_subset = xeol_data.data[:1000,:].compute()
    wavelengths = xeol['x'].data
    x0, y0, sigma = gauss_p0_stack(xeol_subset, wavelengths, xcrop=50)
    p0 = {'x0': np.median(x0), 'y0': np.median(y0), 'sigma': np.median(sigma)}

    # Fit the data
    curvefit = xeol['data'].curvefit(
                                     xeol['data']['x'],
                                     gauss,
                                     p0=p0,
                                     kwargs={'jac': df}
                                     )

    # Extract parameters
    coeff = curvefit['curvefit_coefficients'].data.astype(np.float32)
    cov = curvefit['curvefit_covariance'].data.astype(np.float32)

    # Calculate std errors
    _update_stdout('[5/7] Calculating uncertainties...')
    stdErr = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

    # Fitted data
    coeff = coeff[:,:,np.newaxis]
    fit = gauss(xeol_data, coeff[:,0], coeff[:,1], coeff[:,2])

    # Goodness-of-fit
    _update_stdout('[6/7] Calculating errors...')
    resid = xeol_data - fit
    rmse = np.sqrt((resid**2).mean(dim='x'))
    nrmse = rmse / xeol_data.mean(dim='x')

    # Save output
    nums = xeol['t'].data
    _update_stdout('[7/7] Saving fit parameters...')
    data_vars = {
                 'params': (['t', 'c'], coeff.squeeze()),
                 'params_stderr': (['t', 'c'], stdErr),
                 'gof_rmse': (['t'], rmse.data),
                 'gof_nrmse': (['t'], nrmse.data),
                 }
    coords = dict(t=nums, c=['x0', 'y0', 'sigma'])
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    with ProgressBar():
        out.to_zarr(zfp, mode='a', group='xeol/fit')

    _clear_stdout(2)

    return


def _update_stdout(status):
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[K')
    print(status)

    return


def _clear_stdout(count):
    for i in range(count):
        sys.stdout.write('\033[F')
        sys.stdout.write('\033[K')

    return
