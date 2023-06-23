import numpy as np
import xarray as xr

from sklearn.preprocessing import normalize


def _normalize(vec, axis):
    norm = np.linalg.norm(vec, axis=axis)

    return vec/norm[:,np.newaxis]


def _calcInvRGB_JZT(hkl):
    vec = np.sort(np.abs(_normalize(hkl, axis=1)), axis=1)
    poles = np.array([[0,0,1],[0,1,1],[1,1,1]]).T.astype(float)
    poles = poles / np.linalg.norm(poles, axis=0)
    coefs = np.linalg.inv(poles) @ vec.T
    coefs[coefs < 1E-12] = 0
    rgb = coefs * 1./coefs.max(axis=0)

    return rgb.T


def _calcRGB_JZT(RX, RH, RF, satAngle=None):
    if satAngle is None:
        rr = np.hstack((RX, RH, RF))
        rr = np.degrees(2*np.arctan(np.abs(rr)))
        satAngle = np.nanpercentile(rr, 95)
        satAngle = float(np.format_float_positional(
                                                    satAngle,
                                                    precision=2,
                                                    unique=False,
                                                    fractional=False,
                                                    trim='k'
                                                    ))

    # Combine Euler angles and normalize
    vec3 = np.stack((RX, RH, RF)).T
    mag = np.linalg.norm(vec3, axis=1)
    vec3 /= mag[:, np.newaxis]

    # Scale by angle
    angle = np.degrees(2*np.arctan(mag))
    vec3 *= (angle/satAngle)[:, np.newaxis]

    # Unpack and calculate colors
    cx, cy, cz = vec3.clip(min=-1, max=1).T
    r, g, b = np.zeros((3, cx.shape[0]))

    r[cx>0] += cx[cx>0]
    g[cx<=0] += abs(cx[cx<=0])/2
    b[cx<=0] += abs(cx[cx<=0])/2

    g[cy>0] += cy[cy>0]
    r[cy<=0] += abs(cy[cy<=0])/2
    b[cy<=0] += abs(cy[cy<=0])/2

    b[cz>0] += cz[cz>0]
    r[cz<=0] += abs(cz[cz<=0])/2
    g[cz<=0] += abs(cz[cz<=0])/2

    # Pack into array
    rotRGB = np.stack((r, g, b), axis=1)

    return rotRGB


def map_rgb(zfp, fast_dim, slow_dim):
    zs = xr.open_zarr(zfp, group=f'laue')

    # Load in the hkl and orientation data
    hkl = zs['hkl_float'].load()
    oriMat = zs['oriMat'].load()
    ori = zs['oriXHF'].load()

    # Calculate RGB values
    rotRGB = _calcRGB_JZT(ori[0], ori[1], ori[2])
    invRotRGB = _calcInvRGB_JZT(hkl)

    # Create new xarray Dataset
    data_vars = {
                 'hkl': (['t', 'miller'], hkl.data),
                 'oriMat': (['t', 'rows', 'cols'], oriMat.data),
                 'sampleRGB': (['t', 'color'], rotRGB),
                 'crystalRGB': (['t', 'color'], invRotRGB)
                 }
    coords = {
              't': zs.t.data,
              'miller': zs.miller.data,
              'color': ['red', 'green', 'blue']
              }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Extract parameters and reshape into a 2D map
    # Reshaping by the fastest dimension/inner loop, most likely this is
    # the X motor/width, but depends on the scan setup...
    hkl2D = ds['hkl'].coarsen(t=fast_dim, boundary='pad').construct(t=('ty', 'tx'))
    ori2D = ds['oriMat'].coarsen(t=fast_dim, boundary='pad').construct(t=('ty', 'tx'))
    sample2D = ds['sampleRGB'].coarsen(t=fast_dim, boundary='pad').construct(t=('ty', 'tx'))
    crystal2D = ds['crystalRGB'].coarsen(t=fast_dim, boundary='pad').construct(t=('ty', 'tx'))

    # Make sure the shape is what's expected...
    assert (slow_dim, fast_dim) == sample2D.shape[:2]

    # Create new dataset
    data_vars = {
                 'hkl': (['ty', 'tx', 'miller'], hkl2D.data),
                 'oriMat': (['ty', 'tx', 'rows', 'cols'], ori2D.data),
                 'sampleRGB': (['ty', 'tx', 'color'], sample2D.data),
                 'crystalRGB': (['ty', 'tx', 'color'], crystal2D.data)
                 }
    coords = {
              't': (['ty', 'tx'], sample2D.t.data),
              'tx': sample2D.t.tx, 'ty': sample2D.t.ty,
              'miller': hkl2D.miller.data,
              'color': ['red', 'green', 'blue']
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group=f'maps/laue')

    return
