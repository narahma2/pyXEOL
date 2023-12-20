import math
import numpy as np
import os
import xarray as xr

from glob import glob
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from xml.etree import ElementTree as ET


def process_single(xml_fp, zfp):
    # Get sample info from the XML file
    sampleInfo = _process(xml_fp)

    # Create xarray Dataset
    t = np.arange(0, len(sampleInfo['names']))
    mdir = ['X', 'Y', 'Z']
    sdir = ['X', 'H', 'F']
    miller = ['h', 'k', 'l']

    data_vars = {
                 'laue_folder': sampleInfo['folder'],
                 'laue_fname': (['t'], sampleInfo['names']),
                 'scanNum': sampleInfo['scanNum'],
                 'oriMat': (['t', 'rows', 'cols'], sampleInfo['orientation']),
                 'oriXYZ': (['mdir', 't'], sampleInfo['anglesXYZ']),
                 'oriXHF': (['sdir', 't'], sampleInfo['anglesXHF']),
                 'hkl_float': (['t', 'miller'], sampleInfo['hkl_float']),
                 'hkl_int': (['t', 'miller'], sampleInfo['hkl_int']),
                 'pos_x': (['t'], sampleInfo['sampleX']),
                 'pos_y': (['t'], sampleInfo['sampleY']),
                 'pos_z': (['t'], sampleInfo['sampleZ']),
                 'pos_h': (['t'], sampleInfo['sampleH']),
                 'pos_f': (['t'], sampleInfo['sampleF']),
                 }
    coords = dict(t=t, mdir=mdir, sdir=sdir, miller=miller)
    out = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add in variable attributes
    out['laue_folder'].attrs['label'] = 'Input folder'
    out['laue_fname'].attrs['label'] = 'Input file names'
    out['oriMat'].attrs['label'] = 'Orientation matrix for pattern #0'
    out['oriXYZ'].attrs['label'] = 'Orientation angles in XYZ'
    out['oriXYZ'].attrs['units'] = 'rad'
    out['oriXHF'].attrs['label'] = 'Orientation angles in XHF'
    out['oriXHF'].attrs['units'] = 'rad'
    out['hkl_float'].attrs['label'] = 'Raw HKL values for sample normal'
    out['hkl_int'].attrs['label'] = 'Closest HKL integers for sample normal'
    out['pos_x'].attrs['label'] = 'Sample X position'
    out['pos_x'].attrs['units'] = 'microns'
    out['pos_y'].attrs['label'] = 'Sample Y position'
    out['pos_y'].attrs['units'] = 'microns'
    out['pos_z'].attrs['label'] = 'Sample Z position'
    out['pos_z'].attrs['units'] = 'microns'
    out['pos_h'].attrs['label'] = 'Sample H position'
    out['pos_h'].attrs['units'] = 'microns'
    out['pos_f'].attrs['label'] = 'Sample F position'
    out['pos_f'].attrs['units'] = 'microns'

    # Add in coordinate attributes
    out['t'].attrs['label'] = 'File index'
    out['t'].attrs['units'] = ''
    out['mdir'].attrs['label'] = 'Directions in beamline coordinate system'
    out['mdir'].attrs['units'] = ''
    out['sdir'].attrs['label'] = 'Directions in sample coordinate system'
    out['sdir'].attrs['units'] = ''
    out['miller'].attrs['label'] = 'Miller indices'
    out['miller'].attrs['units'] = ''

    # Write data
    out.to_zarr(zfp, mode='w', group='laue')

    return


def _calcStdLattice(a, b, c, alpha, beta, gamma):
    """
    Taken from LatticeSym.ipf -> setDirectRecip()
    """
    # Degrees -> radians
    alpha, beta, gamma = np.radians((alpha, beta, gamma))

    # Calculate intermediate parameters
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    phi = np.sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
    Vc = (a*b*c) * phi
    pv = (2*np.pi) / Vc

    # Parameters for non-rhombohedral
    a0, a1, a2 = a, 0, 0
    b0, b1, b2 = b*cg, b*sg, 0
    c0, c1, c2 = c*cb, c*(ca-cb*cg)/sg, c*phi/sg

    # Matrix elements
    as0, as1, as2 = (b1*c2-b2*c1)*pv, (b2*c0-b0*c2)*pv, (b0*c1-b1*c0)*pv
    bs0, bs1, bs2 = (c1*a2-c2*a1)*pv, (c2*a0-c0*a2)*pv, (c0*a1-c1*a0)*pv
    cs0, cs1, cs2 = (a1*b2-a2*b1)*pv, (a2*b0-a0*b2)*pv, (a0*b1-a1*b0)*pv

    # Collect into standard reference matrix
    stdLattice = np.array([
                           [as0,bs0,cs0],
                           [as1,bs1,cs1],
                           [as2,bs2,cs2],
                           ])

    return stdLattice


def _yz2hf(y, z):
    cosTheta = np.cos(np.pi/4)
    sinTheta = np.sin(np.pi/4)
    h = y*sinTheta + z*cosTheta
    f = -y*cosTheta + z*sinTheta

    return h, f


def _hklOfNormal(A):
    normal = np.array([[0], [1], [-1]])
    hkl = np.linalg.inv(A) @ normal

    # Normalize
    norm = np.linalg.norm(hkl)
    hkl = hkl / norm

    return hkl


def _hkl2integers(hkl):
    if np.isnan(hkl).any():
        return hkl.flatten()

    # Round hkl
    hkl = np.round(hkl*10, 0).astype(int).flatten()

    # Calculate least common multiple and greatest common divisor
    lcm = math.lcm(*hkl)
    gcd = math.gcd(*hkl)

    # Convert to integers
    if gcd == 1:
        pass
    elif gcd > 1:
        hkl = (hkl / gcd).astype(int)
    elif lcm > 0:
        hkl = (lcm / hkl).astype(int)

    # Round larger numbers to zero (effectively parallel)
    hkl[np.abs(hkl) > 50] = 0

    # Take absolute value and sort
    # FIX: Cubic structures only! Check symmetry operations to generalize...
    hkl = np.sort(np.abs(hkl))

    return hkl


def _symReducedRecipLattice(refLattice, gm, symOps):
    rho = np.eye(3)
    traceMax = -4

    # Iterate through each of the symmetry operations
    for symOp in symOps:
        roti = gm @ np.linalg.inv(rho @ symOp @ refLattice)
        trace = np.trace(roti)

        if trace > traceMax:
            traceMax = trace
            rot = roti

    # Calculate angle (skipping the twinOp as it's not needed for CdTe...)
    cosine = 0.5*(traceMax - 1)
    cosine = np.max((np.min((cosine, 1)), -1))
    angle = np.degrees(np.arccos(cosine))

    return angle, rot


def _axisOfMatrix(rot, squareUp=1):
    # Calculate determinant
    det = np.linalg.det(rot)

    if (abs(det) < 1E-12):
        axis = 0
        return 0
    elif (det<0):
        rot *= -1

    if (squareUp):
        rot = _squareUpMatrix(rot)

    axis = np.zeros(3)
    tr = np.trace(rot)
    if (tr > 0):
        SS = 2*np.sqrt(tr+1)
        qw = SS/4
        axis[0] = rot[2,1] - rot[1,2]
        axis[1] = rot[0,2] - rot[2,0]
        axis[2] = rot[1,0] - rot[0,1]
    elif ((rot[0,0] > rot[1,1]) & (rot[0,0] > rot[2,2])):
        SS = 2*np.sqrt(1 + rot[0,0] - rot[1,1] - rot[2,2])
        qw = (rot[2,1] - rot[1,2]) / SS
        axis[0] = 0.25 * SS
        axis[1] = (rot[0,1] + rot[1,0]) / SS
        axis[2] = (rot[0,2] + rot[2,0]) / SS
    elif (rot[1,1] > rot[2,2]):
        SS = 2*np.sqrt(1 + rot[1,1] - rot[0,0] - rot[2,2])
        qw = (rot[0,2] - rot[2,0]) / SS
        axis[0] = (rot[0,1] + rot[1,0]) / SS
        axis[1] = SS / 4
        axis[2] = (rot[1,2] + rot[2,1]) / SS
    else:
        SS = 2*np.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1])
        qw = (rot[1,0] - rot[0,1]) / SS
        axis[0] = (rot[0,2] + rot[2,0]) / SS
        axis[1] = (rot[1,2] + rot[2,1]) / SS
        axis[2] = SS / 4

    axis = normalize(axis[:, np.newaxis], axis=0)

    angle = 2*np.arccos(qw.clip(min=-1, max=1))
    if (sum(axis)<0):
        axis *= -1
        angle *= -1

    if angle < 0:
        angle += 2*np.pi

    # Convert angle to degrees
    angle = np.mod(np.degrees(angle), 360)

    return angle, axis


def _squareUpMatrix(rot):
    U, S, VH = np.linalg.svd(rot)

    det = np.linalg.det(VH.T @ U.T)
    Idet = np.eye(3)
    Idet[-1,-1] = np.sign(det)

    rot = (VH.T @ Idet @ U.T).T

    return rot


def _getField(node, tag, retType):
    if retType == 'tag':
        return [x.tag for x in node.iter('*') if tag in x.tag]
    elif retType == 'value':
        return [x.text for x in node.iter('*') if tag in x.tag]
    else:
        return None


def _getChildren(node, key, tag, dtype=np.float32):
    val = [
           map(dtype, x.text.split(' '))
           for x in node.find(key).iter('*') if tag in x.tag
           ]

    return val


def _process(fp):
    # Open XML file and get root
    tree = ET.parse(fp)
    root = tree.getroot()

    # Number of positions
    npos = len(root)

    # Scan number (should all be the same!)
    scanNum = int(_getField(root[0], 'scanNum', 'value')[0])

    # File names
    # Sometimes if there are no points the image name isn't saved...so I am
    # building this up manually
    basename = _getField(root[0], 'inputImage', 'value')[0]
    basenum = int(basename.split('_')[-1].split('.h5')[0])
    basename = os.path.basename(basename).split(f'{basenum}.h5')[0]
    names = [f'{basename}{i+basenum}.h5' for i, _ in enumerate(root)]
    folder = os.path.dirname(names[0])

    # Motor positions
    motor_x = np.float32(_getField(root, 'Xsample', 'value'))
    motor_y = np.float32(_getField(root, 'Ysample', 'value'))
    motor_z = np.float32(_getField(root, 'Zsample', 'value'))

    # Calculate position in sample coordinates
    sample_x, sample_y, sample_z = -motor_x, -motor_y, -motor_z
    sample_h, sample_f = _yz2hf(sample_y, sample_z)

    # Indexing key (should be the last element, but just being safe)
    ind_key = _getField(root, 'indexing', 'tag')

    # Initialize lists
    A = [None] * npos
    npatterns = np.zeros(npos, dtype=int)
    hkl_float = np.nan * np.zeros((npos, 3), dtype=np.float32)
    hkl_sym = np.nan * np.zeros((npos, 3), dtype=np.float32)
    hkl_int = np.nan * np.zeros((npos, 3), dtype=np.float32)

    # Populate the hkl direction of the sample normal per point
    normal = np.array([[0],[1],[-1]])

    for i, step in enumerate(root):
        # Skip positions without indexed patterns
        try:
            npatterns[i] = int(step.find(ind_key[i]).attrib['Npatterns'])
            if npatterns[i] == 0:
                continue
        except:
            continue

        # astar/bstar/cstar
        astar = _getChildren(step, ind_key[i], 'astar')
        bstar = _getChildren(step, ind_key[i], 'bstar')
        cstar = _getChildren(step, ind_key[i], 'cstar')

        # Collect into reciprocal matrix
        A[i] = [
                np.reshape((*a, *b, *c), (3,3)).T
                for (a, b, c) in zip(astar, bstar, cstar)
                ]

        # Calculate hkl direction of the sample normal
        hkl = normalize(np.linalg.inv(A[i][0]) @ normal, axis=0)
        hkl_float[i] = hkl.flatten()

        # Get closest hkl integers
        hkl_int[i] = _hkl2integers(hkl_float[i])

    # Get standard lattice (should all be the same)
    latticeParams = _getChildren(root[0], ind_key[0], 'latticeParameters')[0]
    stdLattice = _calcStdLattice(*latticeParams)

    # Load symmetry operations
    symOps_fld = f'{os.path.dirname(os.path.abspath(__file__))}/symOps/'
    symOps = np.load(f'{symOps_fld}/SymmetryOps225.npz')['wData']

    # Loop through the first pattern's reciprocal matrix for symmetry reduction
    rot = np.nan * np.zeros((npos, 3, 3), dtype=np.float32)
    RX, RY, RZ = np.nan*np.zeros((3, npos))
    totalAngles = np.zeros(npos)
    maxAngle = np.inf

    for i in range(npos):
        if A[i] is None:
            continue

        gmi = A[i][0]
        _, rot[i, ...] = _symReducedRecipLattice(stdLattice, gmi, symOps)
        angle, vec3 = _axisOfMatrix(rot[i], squareUp=1)
        vec3 *= np.tan(np.radians(angle)/2)

        if angle <= maxAngle:
            RX[i], RY[i], RZ[i] = vec3.flatten()
            totalAngles[i] = angle

    # Rotation about the X-axis by +45 degrees
    rotFrame = R.from_rotvec([np.pi/4, 0, 0]).as_matrix()

    # Calculate rotations about H/F axes
    Rxhf = (np.linalg.inv(rotFrame) @ np.vstack((RX, RY, RZ))).T
    RH, RF = zip(*Rxhf[:, 1:])

    # Collect sample info
    sampleInfo = {
                  'folder': folder,
                  'names': names,
                  'scanNum': scanNum,
                  'sampleX': sample_x,
                  'sampleY': sample_y,
                  'sampleZ': sample_z,
                  'sampleH': sample_h,
                  'sampleF': sample_f,
                  'orientation': rot,
                  'hkl_float': hkl_float,
                  'hkl_int': hkl_int,
                  'anglesXYZ': np.array((RX, RY, RZ)),
                  'anglesXHF': np.array((RX, RH, RF)),
                  }

    return sampleInfo


def _test():
    git_fld = '/mnt/c/Users/naveed/Documents/GitHub/xsd_mic'
    laue_fld = f'{git_fld}/experiments/2023c1/34IDE/Laue/'
    xml_fp = glob(f'{laue_fld}/recon2D*.xml')
    sampleInfo = _process(xml_fp[0])

    return sampleInfo
