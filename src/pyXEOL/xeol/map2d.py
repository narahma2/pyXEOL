import numpy as np
import xarray as xr


def map_fit(zfp, fast_dim, slow_dim, fit_mode='gauss1'):
    zs = xr.open_zarr(zfp, group=f'xeol/fit_{fit_mode}')

    # Extract parameters and reshape into a 2D map
    # Reshaping by the fastest dimension/inner loop, most likely this is
    # the X motor/width, but depends on the scan setup...
    params = zs['params'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))
    err = zs['params_stderr'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))
    gof_nrmse = zs['gof_nrmse'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))
    gof_rmse = zs['gof_rmse'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))

    # Make sure the shape is what's expected...
    assert (slow_dim, fast_dim) == params.shape[:2]

    # Parameter names
    varNames = zs['c'].data.tolist()

    # Create new dataset
    data_vars1 = {
                  key: (['ty', 'tx'], params[:,:,i].data)
                  for i, key in enumerate(varNames)
                  }
    data_vars2 = {
                  key: (['ty', 'tx'], err[:,:,i].data)
                  for i, key in enumerate(varNames)
                  }
    data_vars3 = {
                  'nrmse': (['ty', 'tx'], gof_nrmse.data),
                  'rmse': (['ty', 'tx'], gof_rmse.data)
                  }
    coords = {
              't': (['ty', 'tx'], params.t.data),
              'tx': params.t.tx, 'ty': params.t.ty
              }
    out1 = xr.Dataset(data_vars=data_vars1, coords=coords)
    out2 = xr.Dataset(data_vars=data_vars2, coords=coords)
    out3 = xr.Dataset(data_vars=data_vars3, coords=coords)

    # Save output
    out1.to_zarr(zfp, mode='w', group=f'maps/fit_{fit_mode}')
    out2.to_zarr(zfp, mode='w', group=f'maps/fit_err_{fit_mode}')
    out3.to_zarr(zfp, mode='w', group=f'maps/fit_gof_{fit_mode}')

    return


def map_sumInt(zfp, lb, ub, fast_dim, slow_dim):
    zs = xr.open_zarr(zfp, group='xeol')

    # Extract spectra and reshape into a 2D map
    # Reshaping by the fastest dimension/inner loop, most likely this is
    # the X motor/width, but depends on the scan setup...
    spectra = zs['data'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))

    # Make sure the shape is what's expected...
    assert (slow_dim, fast_dim) == spectra.shape[:2]

    # Get masked indices
    mask = np.logical_and(zs['x'] > lb, zs['x'] < ub)
    spectra = spectra[:,:,mask]

    # Sum over the wavelengths
    sumInt = spectra.sum(dim='x')

    # Create new dataset
    data_vars = {
                 f'sum_{lb}to{ub}': (['ty', 'tx'], sumInt.data),
                 }
    coords = {
              't': (['ty', 'tx'], spectra.t.data),
              'tx': spectra.t.tx, 'ty': spectra.t.ty
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='a', group='maps/sumInt')

    return


def map_stats(zfp, fast_dim, slow_dim):
    zs = xr.open_zarr(zfp, group='xeol')

    # Extract spectra/snr and reshape into a 2D map
    # Reshaping by the fastest dimension/inner loop, most likely this is
    # the X motor/width, but depends on the scan setup...
    spectra = zs['data'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))
    snr = zs['snr'].coarsen(t=fast_dim).construct(t=('ty', 'tx'))

    # Make sure the shape is what's expected...
    assert (slow_dim, fast_dim) == spectra.shape[:2]

    # Smooth data
    smooth = spectra.rolling(x=10, center=True).mean().load()

    # Calculate peaks
    peaks = smooth.max(dim='x')

    # Calculate peak centers
    centers_px = smooth.argmax(dim='x')
    centers_nm = smooth['x'][centers_px]

    # Create new dataset
    data_vars = {
                 'snr': (['ty', 'tx'], snr.data),
                 'peaks': (['ty', 'tx'], peaks.data),
                 'peakCenters': (['ty', 'tx'], centers_nm.data),
                 }
    coords = {
              't': (['ty', 'tx'], spectra.t.data),
              'tx': spectra.t.tx, 'ty': spectra.t.ty
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/stats')

    return
