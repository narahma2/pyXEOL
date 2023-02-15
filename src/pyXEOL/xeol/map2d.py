import numpy as np
import xarray as xr


def map_fit(zfp):
    zs = xr.open_zarr(zfp, group='xeol/fit')

    # Extract parameters and reshape into a 2D map
    shape = np.repeat(int(np.sqrt(zs['params'].shape[0])), 2)
    x0 = zs['params'][:,0].data.reshape(shape)
    y0 = zs['params'][:,1].data.reshape(shape)
    sigma = zs['params'][:,2].data.reshape(shape)
    t = zs['t'].data.reshape(shape)

    # Create new dataset
    data_vars = {
                 'x0': (['tx', 'ty'], x0),
                 'y0': (['tx', 'ty'], y0),
                 'sigma': (['tx', 'ty'], sigma),
                 }
    coords = {'t': (['tx', 'ty'], t)}
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/fit')

    return


def map_sumInt(zfp, lb, ub):
    zs = xr.open_zarr(zfp, group='xeol')

    # Extract parameters and reshape into a 2D map as needed
    shape = np.repeat(int(np.sqrt(zs['data'].shape[0])), 2)
    spectra = zs['data']
    t = zs['t'].data.reshape(shape)

    # Get masked indices
    mask = np.logical_and(zs['x'] > lb, zs['x'] < ub)
    spectra = spectra[:, mask]

    # Sum over the wavelengths and reshape
    sumInt = spectra.sum(dim='x').data.reshape(shape)

    # Create new dataset
    data_vars = {
                 f'sum_{lb}to{ub}': (['tx', 'ty'], sumInt),
                 }
    coords = {'t': (['tx', 'ty'], t)}
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/sumInt')

    return
