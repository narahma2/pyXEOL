import h5py
import numpy as np
import periodictable as pt
import xarray as xr

from glob import glob


def process(inp, zfp, elements, quant=None, beamline='2idd'):
    # Retrieve XRF data
    if beamline == '2idd':
        xrf = _load_xrf_2idd(inp, elements, quant=quant)

    # Output data as a 2D map
    if xrf is not None:
        _map_xrf(zfp, xrf)

    return


def _load_xrf_2idd(h50_fp, elements, quant=None):
    # Initialize output
    xrf = {}

    # Load h5 file
    with h5py.File(h50_fp, 'r') as hf:
        # Fit type to use
        use_type = 'MAPS/XRF_Analyzed/NNLS'

        # Channel names
        path = f'{use_type}/Channel_Names'
        channels = hf[path][()].astype('U13')

        # Strip out the edges
        channels = np.array([x.split('_')[0] for x in channels])

        # Sort the requested elements by atomic number
        el_num = [getattr(pt, x).number for x in elements]
        order = np.argsort(el_num)
        elements = np.array(elements)[order].tolist()

        # Load XRF data
        xrf_all = hf[f'{use_type}/Counts_Per_Sec']

        # Check that a full 2D array was scanned
        if (xrf_all.shape[0] > 1) and (xrf_all.shape[1] > 1):
            pass
        else:
            return None

        # Get requested channels
        for el in elements:
            el_ind = np.where(channels == el)[0][0]

            # Crop last two columns
            xrf_el = xrf_all[el_ind,:,:-2].squeeze()

            # Populate output
            xrf[el] = xrf_el

    return xrf


def _map_xrf(zfp, xrf):
    # Check if XEOL maps exist to verify shape
    maps_zfp = glob(f'{zfp}/maps/xeol/*')

    if len(maps_zfp):
        store = maps_zfp[0].split('/maps')[0]
        group = maps_zfp[0].split('.zarr/')[1]
        zs = xr.open_zarr(store, group=group)
        shape = [zs[x].shape for x in zs.keys()][0]
        t = zs.t.data
        tx = zs.tx.data
        ty = zs.ty.data

        # Make sure the shape is what's expected...
        assert shape == next(iter(xrf.items()))[1].shape
    else:
        shape = next(iter(xrf.items()))[1].shape
        t = np.arange(np.prod(shape)).reshape(shape)
        tx = np.arange(shape[1])
        ty = np.arange(shape[0])

    # Create new dataset
    data_vars = {key: (['ty', 'tx'], value) for (key, value) in xrf.items()}
    coords = {
              't': (['ty', 'tx'], t),
              'tx': tx, 'ty': ty
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group='maps/xrf')

    return
