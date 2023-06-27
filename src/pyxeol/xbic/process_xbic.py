import h5py
import numpy as np
import xarray as xr

from glob import glob


def process(inp, zfp, sens=None, beamline='2idd'):
    # Retrieve XBIC data
    if beamline == '2idd':
        xbic = _load_xbic_2idd(inp, sens=sens)

    # Output data as a 2D map
    if xbic is not None:
        _map_xbic(zfp, xbic)

    return


def _load_xbic_2idd(h50_fp, sens=None):
    # Load h5 file
    with h5py.File(h50_fp, 'r') as hf:
        # Check if XBIC was done, otherwise skip
        try:
            scalers = hf['MAPS/scaler_names'][()].astype('U13')
        except:
            return None

        # Get the upstream/downstream
        usic_ind = np.where(scalers == 'US_IC')[0]
        dsic_ind = np.where(scalers == 'DS_IC')[0]
        dsic = hf['MAPS/scalers'][dsic_ind][()]
        usic = hf['MAPS/scalers'][usic_ind][()]

        # Calculate XBIC and crop out the last two columns
        xbic = np.divide(dsic, usic, out=np.zeros_like(usic), where=usic!=0)
        xbic = xbic.squeeze()[:,:-2]

    return xbic


def _map_xbic(zfp, xbic):
    # Check if XEOL maps exist to verify shape
    maps_zfp = glob(f'{zfp}/maps/**/*')

    if len(maps_zfp):
        store = maps_zfp[0].split('/maps')[0]
        group = maps_zfp[0].split('.zarr/')[1]
        zs = xr.open_zarr(store, group=group)
        shape = [zs[x].shape for x in zs.keys()][0]
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
                 f'xbic': (['ty', 'tx'], xbic),
                 }
    coords = {
              't': (['ty', 'tx'], t),
              'tx': tx, 'ty': ty
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/xbic')

    return
