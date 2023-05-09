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

from pyxeol.specfun import (
                            baseline_arPLS,
                            df,
                            df2,
                            gauss,
                            gauss2,
                            gauss_p0,
                            gauss_p0_stack,
                            sigma2fwhm,
                            )
from pyxeol.misc import create_folder


def process_stack_dask(
                       in_fld,
                       wavelengths,
                       zfp,
                       load=True,
                       fit=True,
                       fit_mode='gauss1',
                       wl_crop=None
                       ):
    # Load in files and sort (just in case)
    fn = glob(f'{in_fld}/*')

    # Continue if no files were loaded
    if not bool(fn):
        return 0

    if '.tif' not in fn[0]:
        im = None
        ind = np.argsort([int(x.split('_')[-1]) for x in fn])
        fn = np.array(fn)[ind].tolist()
    else:
        im = f'{in_fld}/*.tif'
        fn = glob(f'{in_fld}/*.tif')

    # Process results
    print(f'{in_fld.split("/")[-1]} processing...')
    if fit:
        if fit_mode == 'gauss2':
            steps = 8
        else:
            steps = 7

        print(f'[0/{steps}] Initiating...')
        if load:
            _pipeline_image2spec(fn, im, zfp, wavelengths, steps, wl_crop)
        _pipeline_specfit(zfp, steps, fit_mode)
    else:
        steps = 3
        print(f'[0/{steps}] Initiating...')
        if load:
            _pipeline_image2spec(fn, im, zfp, wavelengths, steps, wl_crop)
            _clear_stdout(2)

    _update_stdout(f'{in_fld.split("/")[-1]} finished!')

    return 1


def process_single(
                   fp,
                   wavelengths,
                   x=None,
                   bg_corr=False,
                   wl_crop=None,
                   fit_mode='gauss1',
                   plot=True,
                   figsize=None
                   ):
    if '.tif' in fp:
        raw2D = np.array(Image.open(fp)).astype(np.float32)
    else:
        raw2D = xr.open_dataset(fp)['array_data'].data.astype(np.float32)
        raw2D = raw2D.squeeze()

    # Select X point (for data saved as lines)
    if x is not None:
        raw2D = raw2D[x,:,:].squeeze()

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
    x0, y0, sigma0 = gauss_p0(ds, wavelengths, xcrop=50)

    # Fit the data, output guesses if it doesn't work
    try:
        if fit_mode.lower() == 'gauss1':
            model, df = (gauss, df)
            p0 = [x0, y0, sigma0]
            bnds = (
                    (wavelengths.min(), wavelengths.max()),
                    (0, np.inf),
                    (0, np.inf),
                    )

        elif fit_mode.lower() == 'gauss2':
            model, df = (gauss2, df2)
            p0 = [x0, y0, sigma0, x0, y0, sigma0]
            bnds = (
                    (x0-100, 0, 0, x0-100, 0, 0),
                    (x0+100, np.inf, np.inf, x0+100, np.inf, np.inf),
                    )

        # Try fitting the dataset
        pFit, _ = curve_fit(model, wavelengths, ds, p0=p0, jac=df, bounds=bnds)
        status = 1

    except:
        if fit_mode.lower() == 'gauss1':
            pFit = [x0, y0, sigma0]
        elif fit_mode.lower() == 'gauss2':
            pFit = [x0, y0, sigma0, x0, y0, sigma0]

        status = 0
        pass

    # Plot results (if requested)
    if plot:
        if figsize is not None:
            figsize = (12, 12)

        vmin = np.percentile(raw2D, 1)
        vmax=  np.percentile(raw2D, 99)

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        #plt.subplots_adjust(hspace=0.01)
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
                       model(wavelengths, *pFit),
                       linestyle='--',
                       color='r',
                       alpha=0.8,
                       label='Fit',
                       )
        ax[1].legend()

        for x in pFit[::3]:
            ax[1].axvline(x, linestyle='--', color='gray', alpha=0.8)

        ax[1].set_xlabel('Wavelength (nm)')
        ax[1].set_ylabel('Intensity (a.u.)')

        if fit_mode.lower() == 'gauss1':
            string = (
                      f'Center = {pFit[0]:0.2f} nm\n'
                      f'Peak = {pFit[1]:0.2f}\n'
                      f'FWHM = {sigma2fwhm(pFit[2]):0.2f} nm'
                      )
        elif fit_mode.lower() == 'gauss2':
            string = (
                      f'Centers = ({pFit[0]:0.2f}, {pFit[3]:0.2f}) nm\n'
                      f'Peaks = ({pFit[1]:0.2e}, {pFit[4]:0.2e})\n'
                      f'FWHMs = ({sigma2fwhm(pFit[2]):0.2f}, ' +
                               f'{sigma2fwhm(pFit[5]):0.2f}) nm'
                      )
        ax[1].text(
                   0.03, 0.7,
                   string,
                   transform=ax[1].transAxes,
                   fontsize='small'
                   )

        plt.tight_layout()
        plt.show()

    return (status, pFit)


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

    # Re-chunk if needed (for data saved as lines instead of as points)
    if raw.chunksize[0] != 1:
        raw = raw.rechunk((1, -1, -1))

    # Crop down images
    if wl_crop is not None:
        raw = raw[:,:,wl_crop:-wl_crop]

    # Calculate SNR (using 99th percentile)
    noise = np.nanstd(raw[:,:100,:100], axis=(1,2))
    signal = raw.map_blocks(np.nanpercentile, 99, drop_axis=(1,2))
    snr = signal / noise

    # Baseline (scalar subtraction)
    bl = da.median(raw[:,::5,-100:], axis=(1,2))
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
    data_vars = {
                 'folder': fn[0].split('/')[-2],
                 'files': names,
                 }
    #coords = dict(t=nums)
    out1 = xr.Dataset(data_vars=data_vars)
    out1.to_zarr(zfp, mode='w', group='/')

    nums = np.arange(0, ds.shape[0])
    data_vars = {
                 'bg': (['x'], bg),
                 'snr': (['t'], snr),
                 'data': (['t', 'x'], ds),
                 }
    coords = dict(x=wavelengths, t=nums)
    out2 = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add in variable attributes
    out2['bg'].attrs['label'] = 'Vector background'
    out2['snr'].attrs['label'] = 'Signal-to-noise raio (99th percentile)'
    out2['data'].attrs['label'] = 'Measured data'

    # Add in coordinate attributes
    out2['x'].attrs['label'] = 'Wavelength'
    out2['x'].attrs['units'] = 'nm'
    out2['t'].attrs['label'] = 'File number'
    out2['t'].attrs['units'] = ''

    with ProgressBar():
        out2.to_zarr(zfp, mode='w', group='xeol')

    return


def _pipeline_specfit(zfp, steps, fit_mode='gauss1'):
    _clear_stdout(1)

    # Load in the processed spectra data
    _update_stdout(f'[4/{steps}] Initiating fitting...')
    xeol = xr.open_zarr(zfp, group='xeol', drop_variables=['bg', 'snr'])

    # Gaussian parameter estimates and bounds using a random sample of the data
    numPts = xeol['t'].shape[0]
    rand_ind = np.random.randint(0, numPts, int(0.1*numPts))
    xeol_subset = xeol.data[rand_ind,:].compute()
    wavelengths = xeol['x'].data
    x0, y0, sigma0 = gauss_p0_stack(xeol_subset, wavelengths, xcrop=50)

    if fit_mode == 'gauss1':
        p0 = {
              'x1': np.nanmedian(x0),
              'y1': np.nanmedian(y0),
              'sigma1': np.nanmedian(sigma0)
              }
        bounds = {
                  'x1': (wavelengths[0], wavelengths[-1]),
                  'y1': (0, 1.5*np.nanmax(y0)),
                  'sigma1': (
                             np.nanmax((0,np.min(sigma0)/2)),
                             1.5*np.nanmax(sigma0)
                             )
                  }
        varNames = ['x1', 'y1', 'sigma1']
        model = gauss
        use_df = df

    elif fit_mode == 'gauss2':
        p0 = {
              'x1': np.nanmedian(x0),
              'y1': np.nanmedian(y0),
              'sigma1': np.nanmedian(sigma0),
              'x2': np.nanmedian(x0),
              'y2': np.nanmedian(y0),
              'sigma2': np.nanmedian(sigma0)
              }
        bounds = {
                  'x1': (wavelengths[0], wavelengths[-1]),
                  'y1': (0, 1.5*np.nanmax(y0)),
                  'sigma1': (
                             np.nanmax((0,np.min(sigma0)/2)),
                             1.5*np.nanmax(sigma0)
                             ),
                  'x2': (wavelengths[0], wavelengths[-1]),
                  'y2': (0, 1.5*np.nanmax(y0)),
                  'sigma2': (
                             np.nanmax((0,np.min(sigma0)/2)),
                             1.5*np.nanmax(sigma0)
                             )
                  }
        varNames = ['x1', 'y1', 'sigma1', 'x2', 'y2', 'sigma2']
        model = gauss2
        use_df = df2

    # Fit the data
    curvefit = xeol.curvefit(
                             xeol['x'],
                             model,
                             p0=p0,
                             bounds=bounds,
                             kwargs={'jac': use_df},
                             allow_failures=True
                             )

    # Extract parameters
    coeff = curvefit['data_curvefit_coefficients'].data.astype(np.float32)
    cov = curvefit['data_curvefit_covariance'].data.astype(np.float32)

    # Calculate std errors
    _update_stdout(f'[5/{steps}] Calculating uncertainties...')
    stdErr = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

    # Calculate fit profiles
    if fit_mode == 'gauss1':
        fit = gauss(xeol['data'].data, *coeff.T[:,:,np.newaxis])
        c_units = ['nm', 'a.u.', 'nm']
    elif fit_mode == 'gauss2':
        fit = gauss2(xeol['data'].data, *coeff.T[:,:,np.newaxis])
        c_units = ['nm', 'a.u.', 'nm', 'nm', 'a.u.', 'nm']

    # Goodness-of-fit
    _update_stdout(f'[6/{steps}] Calculating errors...')
    resid = xeol['data'] - fit
    rmse = np.sqrt((resid**2).mean(dim='x')).data
    nrmse = rmse / xeol['data'].mean(dim='x').data

    # Save output
    nums = xeol['t'].data
    _update_stdout(f'[7/{steps}] Saving fit parameters...')
    data_vars = {
                 'params': (['t', 'c'], coeff.squeeze()),
                 'params_stderr': (['t', 'c'], stdErr),
                 'gof_rmse': (['t'], rmse),
                 'gof_nrmse': (['t'], nrmse),
                 }
    coords = dict(t=nums, c=varNames, x=wavelengths)
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add in variable attributes
    out['params'].attrs['label'] = 'Vector background'
    out['params_stderr'].attrs['label'] = 'Measured data'
    out['gof_rmse'].attrs['label'] = 'Goodness of fit (RMSE)'
    out['gof_nrmse'].attrs['label'] = 'Goodness of fit (NRMSE)'

    # Add in coordinate attributes
    out['x'].attrs = xeol['x'].attrs
    out['t'].attrs = xeol['t'].attrs
    out['c'].attrs['label'] = f'{fit_mode} fit parameters'
    out['c'].attrs['units'] = c_units

    #write_job = out.to_zarr(zfp, mode='a', group=f'xeol/fit_{fit_mode}',
    #                        compute=False)

    with ProgressBar():
        out.to_zarr(zfp, mode='a', group=f'xeol/fit_{fit_mode}')

    # Re-order the gauss2 parameters by peak position
    if fit_mode == 'gauss2':
        _update_stdout(f'[8/{steps}] Re-ordering gauss2 parameters...')
        out = xr.open_zarr(zfp, group=f'xeol/fit_{fit_mode}').load()

        # Get new indices that need to be re-ordered
        reind = out['params'].sel(c='x1') > out['params'].sel(c='x2')

        # Re-order
        out['params'][reind,:] = out['params'][reind,:].roll(c=3)
        out['params_stderr'][reind,:] = out['params_stderr'][reind,:].roll(c=3)

        # Re-save output
        out.to_zarr(zfp, mode='a', group=f'xeol/fit_{fit_mode}')

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
