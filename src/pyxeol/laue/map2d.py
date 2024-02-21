import numpy as np
import xarray as xr

from skimage import color
from skimage.measure import find_contours, label, regionprops


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


def _closestRGB(crystal2D):
    colors = crystal2D.data.reshape((-1, 3))
    red = np.array([1,0,0])
    green = np.array([0,1,0])
    blue = np.array([0,0,1])

    closest = np.zeros_like(colors)

    for i, c in enumerate(colors):
        vec = [np.sqrt(np.sum((c - rgb)**2)) for rgb in (red, green, blue)]
        ind = np.argmin(vec)
        closest[i,:] = [red, green, blue][ind]

    closest = closest.reshape(crystal2D.shape)

    # Calculate percentages of red/green/blue bins
    perc = closest.sum(axis=(0,1)) / closest.sum()

    return closest, perc


def _segmentation(crystal2D):
    # Segmentation on the Laue sample colors
    laue_gray = color.rgb2gray(np.round(crystal2D, 1))
    laue_hue = np.round(laue_gray, 5)
    unique_vals = np.unique(laue_hue)

    grain_label = []
    grain_coords = []
    grain_props = []
    grain_contour = []
    grain_image = np.nan * np.zeros_like(laue_hue)

    i = -1
    for x in unique_vals:
        # Label each value and remove smaller regions
        lbl = label(laue_hue == x)
        regions = regionprops(lbl)

        for r in regions:
            if r.area < 10:
                continue

            i += 1
            grain_label.append(i)
            grain_coords.append(r.coords)
            grain_props.append(r)
            grain_image[r.coords[:,0], r.coords[:,1]] = i
            im0 = np.zeros_like(laue_hue)
            im0[r.coords[:,0], r.coords[:,1]] = 1
            grain_contour.append(np.fliplr(find_contours(im0)[0]))

    # Closest RGB colors for crystal orientation
    closestRGB, percRGB = _closestRGB(crystal2D)

    # Masks for each direction
    mask001 = np.all(closestRGB == [1,0,0], axis=2).astype(np.float32)
    mask101 = np.all(closestRGB == [0,1,0], axis=2).astype(np.float32)
    mask111 = np.all(closestRGB == [0,0,1], axis=2).astype(np.float32)
    mask001[mask001 == 0] = np.nan
    mask101[mask101 == 0] = np.nan
    mask111[mask111 == 0] = np.nan

    # Mask for each grain
    grainRGB = np.array([
                         closestRGB[c[:,0], c[:,1]].mean(axis=0)
                         for c in grain_coords
                         ])
    grain001 = np.all(grainRGB == [1,0,0], axis=1)
    grain101 = np.all(grainRGB == [0,1,0], axis=1)
    grain111 = np.all(grainRGB == [0,0,1], axis=1)

    # Pad contours to keep numPts size the same
    numPts = [len(x) for x in grain_contour]
    padding = np.max(numPts) - np.array(numPts)
    grain_contour = [
                     np.pad(
                            x,
                            pad_width=((0,padding[i]), (0,0)),
                            constant_values=np.nan
                            )
                     for i, x in enumerate(grain_contour)
                     ]

    # Package data
    out = {
           'all_grains': grain_image,
           'contour': np.array(grain_contour),
           'grains001': grain001,
           'grains101': grain101,
           'grains111': grain111,
           'mask001': mask001,
           'mask101': mask101,
           'mask111': mask111,
           }

    return out


def map_grains(zfp, fast_dim, slow_dim, satAngle=None):
    zs = xr.open_zarr(zfp, group=f'laue')

    # Load in the fwhm, hkl, and orientation data
    fwhm = zs['fwhm'].load()
    hkl = zs['hkl_float'].load()
    oriMat = zs['oriMat'].load()
    ori = zs['oriXHF'].load()

    # Calculate RGB values
    rotRGB = _calcRGB_JZT(ori[0], ori[1], ori[2], satAngle=satAngle)
    invRotRGB = _calcInvRGB_JZT(hkl)

    # Create new xarray Dataset
    data_vars = {
                 'fwhm': (['t'], fwhm.data),
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
    reshape = lambda x: x.coarsen(t=fast_dim, boundary='pad') \
                         .construct(t=('ty', 'tx'))
    fwhm2D = reshape(ds['fwhm'])
    hkl2D = reshape(ds['hkl'])
    ori2D = reshape(ds['oriMat'])
    sample2D = reshape(ds['sampleRGB'])
    crystal2D = reshape(ds['crystalRGB'])

    # Make sure the shape is what's expected...
    assert (slow_dim, fast_dim) == sample2D.shape[:2]

    # Run segmentation
    grains = _segmentation(crystal2D)

    # Create new dataset
    data_vars = {
                 'fwhm': (['ty', 'tx'], fwhm2D.data),
                 'hkl': (['ty', 'tx', 'miller'], hkl2D.data),
                 'oriMat': (['ty', 'tx', 'rows', 'cols'], ori2D.data),
                 'sampleRGB': (['ty', 'tx', 'color'], sample2D.data),
                 'crystalRGB': (['ty', 'tx', 'color'], crystal2D.data),
                 'allGrains': (['ty', 'tx'], grains['all_grains']),
                 'mask001': (['ty', 'tx'], grains['mask001']),
                 'mask101': (['ty', 'tx'], grains['mask101']),
                 'mask111': (['ty', 'tx'], grains['mask111']),
                 'grains001': (['grain'], grains['grains001']),
                 'grains101': (['grain'], grains['grains101']),
                 'grains111': (['grain'], grains['grains111']),
                 'contour': (['grain', 'pt', 'xy'], grains['contour']),
                 }
    coords = {
              't': (['ty', 'tx'], sample2D.t.data),
              'tx': sample2D.t.tx, 'ty': sample2D.t.ty,
              'miller': hkl2D.miller.data,
              'color': ['red', 'green', 'blue'],
              'grain': np.arange(np.nanmax(grains['all_grains'])+1),
              }
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save output
    out.to_zarr(zfp, mode='w', group=f'maps/laue')

    return
