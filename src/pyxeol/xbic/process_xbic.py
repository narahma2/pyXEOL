import h5py
import numpy as np
import xarray as xr

from glob import glob


def process(inp, zfp, sens=None, beamline='2idd'):
    # Retrieve XBIC data
    if beamline == '2idd':
        usic, xbic = _load_xbic_2idd(inp, sens=sens)

    # Output data as a 2D map
    if xbic is not None:
        _map_xbic(zfp, usic, xbic)

    return


def _load_xbic_2idd(h50_fp, sens=None):
    # Load h5 file
    with h5py.File(h50_fp, 'r') as hf:
        # Check if XBIC was done, otherwise skip
        try:
            scalers = hf['MAPS/Scalers/Names'][()].astype('U13')
        except:
            return None

        # Get the upstream/downstream ion chamber readout
        usic_ind = np.where(scalers == 'US_IC')[0]
        dsic_ind = np.where(scalers == 'DS_IC')[0]
        dsic = hf['MAPS/Scalers/Values'][dsic_ind][0][:,:-2].squeeze()
        usic = hf['MAPS/Scalers/Values'][usic_ind][0][:,:-2].squeeze()

        # Calculate XBIC and crop out the last two columns
        xbic = np.divide(dsic, usic, out=np.zeros_like(usic), where=usic!=0)
        xbic = xbic.squeeze()

    return usic, xbic


def _map_xbic(zfp, usic, xbic):
    # Check if scalars exist to verify shape
    scalars = glob(f'{zfp}/scalars')

    if len(scalars):
        zs = xr.open_zarr(zfp, group='scalars')
        shape = zs.t.shape
        t = zs.t.data
        tx = zs.tx.data
        ty = zs.ty.data

        # Make sure the shape is what's expected...
        assert shape == xbic.shape
    else:
        t = np.arange(np.prod(xbic.shape)).reshape(xbic.shape)
        tx = np.arange(xbic.shape[1])
        ty = np.arange(xbic.shape[0])

    # Create new dataset
    data_vars = {
                 f'usic': (['ty', 'tx'], usic),
                 f'xbic': (['ty', 'tx'], xbic)
                 }
    coords = {
              't': (['ty', 'tx'], t),
              'tx': tx, 'ty': ty
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/xbic')

    return
