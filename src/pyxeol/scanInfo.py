import numpy as np
import xarray as xr

from pyxeol.readMDA import scanDim


def compile_scalars(mda, scalars, zfp, beamline='2idd'):
    # Loop through all requested scalars
    data = []
    metadata = []
    for key in scalars.keys():
        # Check if requested scalar is a motor position
        if 'pos_' in key:
            xy = _motorXY(mda, scalars.get(key), zfp)
            if xy is not None:
                data.append((key, xy))
            else:
                return 0
        # Desginate as metadata
        elif 'meta_' in key:
            metadata.append((key.split('meta_')[-1], scalars.get(key)))
        # Otherwise it's another data variable
        else:
            data.append((key, scalars.get(key)))

    # Crop the last two columns for 2-IDD (saving error?)
    if beamline == '2idd':
        crop = -2
    else:
        crop = None

    # Create new dataset
    dimensions = mda[0]['dimensions']
    t = np.arange(np.prod(dimensions)).reshape(dimensions)[:, :crop]
    tx = np.arange(dimensions[1])[:crop]
    ty = np.arange(dimensions[0])
    coords = {
              't': (['ty', 'tx'], t),
              'tx': tx, 'ty': ty
              }

    # Fix the variables below to correctly load from data...
    data_vars = {
                 x[0]: (['ty', 'tx'], x[1][:, :crop])
                 for x in data
                 }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add in the metadata
    [out.__setitem__(x[0], x[1]) for x in metadata]

    # Save output
    out.to_zarr(zfp, mode='w', group='scalars')

    return 1


def _motorXY(mda, sname, zfp):
    # Load in scan dimensions
    dimensions = mda[0]['dimensions']

    # Locate the scaler
    inds = [
            [(d1, d2) for d2, x in enumerate(y.p) if x.name == sname]
            for d1, y in enumerate(mda) if type(y) == scanDim
            ]
    inds = np.ravel([x for x in inds if x != []])

    # Get values
    try:
        val = mda[inds[0]].p[inds[1]].data
    except:
        return None

    # Reshape the array to match the scan dimensions
    if np.ndim(val) == 2:
        val = np.array(val)
    else:
        val = np.tile(val, (dimensions[1], 1)).T

    # Check that the dimensions match
    if not np.all(val.shape == tuple(dimensions)):
        return None

    return val
