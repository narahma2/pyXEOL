import numpy as np
import os

from glob import glob
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from xml.etree import ElementTree as ET

# Environment version info:
# numpy: 1.21.5
# igor: 0.3
# matplotlib: 3.5.2
# scipy: 1.9.1
# scikit-learn: 1.0.2


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


def _hkl2rgb(hkl):
    if hkl is None:
        return np.array(3*[np.nan])

    vec = abs(normalize(hkl, axis=0))
    poles = np.array([[0,0,1],[0,1,1],[1,1,1]]).T.astype(float)
    poles = normalize(poles, axis=0)
    coefs = np.linalg.inv(poles) @ vec
    coefs[coefs < 1E-12] = 0
    rgb = coefs * 1./coefs.max()

    return rgb.flatten()


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


def _getField(root, tag, retType):
    if retType == 'tag':
        return [x.tag for x in root.iter('*') if tag in x.tag]
    elif retType == 'value':
        return [x.text for x in root.iter('*') if tag in x.tag]
    else:
        return None


def process(fp):
    # Open XML file and get root
    tree = ET.parse(fp)
    root = tree.getroot()

    # Number of positions
    npos = len(root)

    # Motor positions
    motor_x = np.float32(_getField(root, 'Xsample', 'value'))
    motor_y = np.float32(_getField(root, 'Ysample', 'value'))
    motor_z = np.float32(_getField(root, 'Zsample', 'value'))

    # Calculate position in sample coordinates
    sample_x, sample_y, sample_z = -motor_x, -motor_y, -motor_z
    sample_h, sample_f = _yz2hf(sample_y, sample_z)

    # Indexing key (should be the last element, but just being safe)
    ind_key = _getField(root, 'indexing', 'tag')

    # Number of patterns per position
    npatterns = [
                 int(root[i].find(ind_key[i]).attrib['Npatterns'])
                 for i in range(npos)
                 ]

    # Initialize lists
    A = [None] * npos
    hkl_grain = [None] * npos
    Qxyz = [None] * npos
    hkl = [None] * npos
    xy = [None] * npos

    # Populate the hkl direction of the sample normal per point
    nhat = normalize([[0],[1],[-1]], axis=0)

    for i, step in enumerate(root):
        # Skip positions without indexed patterns
        if npatterns[i] == 0:
            continue

        # astar/bstar/cstar
        astar = [
                 x[0].text.split(' ') for x in step.find(ind_key[i]).iter('*')
                 if 'recip' in x.tag
                 ]
        bstar = [
                 x[1].text.split(' ') for x in step.find(ind_key[i]).iter('*')
                 if 'recip' in x.tag
                 ]
        cstar = [
                 x[2].text.split(' ') for x in step.find(ind_key[i]).iter('*')
                 if 'recip' in x.tag
                 ]

        # hkl of each of the indexed points as a list of lists
        tmp = [
               [
                list(map(int, x[n].text.split(' ')))
                for x in step.find(ind_key[i]).iter('*')
                if 'hkl' in x.tag
                ]
               for n in range(3)
               ]

        # Number of points used in each pattern
        npt = [len(x) for x in tmp[0]]

        # Convert hkl's into tuples corresponding to each point in each pattern
        hkl[i] = [
                  [(tmp[0][p][q],tmp[1][p][q],tmp[2][p][q]) for q in range(npt[p])]
                  for p in range(npatterns[i])
                  ]

        # Indexed peaks used for fitting
        peaks_ind = [
                     list(map(int, x[3].text.split(' ')))
                     for x in step.find(ind_key[i]).iter('*')
                     if 'hkl' in x.tag
                     ]

        # Detector, peaksXY, and Qx/Qy keys
        dt = [x.tag for x in root[i] if 'detector' in x.tag][0]
        pk = [x.tag for x in root[i].find(dt) if 'peaksXY' in x.tag][0]
        qx = [x.tag for x in root[i].find(dt).find(pk) if 'Qx' in x.tag][0]
        qy = [x.tag for x in root[i].find(dt).find(pk) if 'Qy' in x.tag][0]
        qz = [x.tag for x in root[i].find(dt).find(pk) if 'Qz' in x.tag][0]

        # Get all the Qx/Qy/Qz values
        Qxyz = [
                np.float32(root[i].find(dt).find(pk).find(x).text.split(' '))
                for x in (qx, qy, qz)
                ]

        # Convert into a NumPy array
        Qxyz = np.array(Qxyz).T

        # Get used points only
        Qxyz_used = [Qxyz[p] for p in peaks_ind]

        # Collect into reciprocal matrix
        A[i] = [
                np.stack(list(map(np.float32, (a, b, c)))).T
                for (a, b, c) in zip(astar, bstar, cstar)
                ]

        # Calculate the Qhat vector manually (just to check)
        Qxyz_check = [
                      normalize((Ai @ np.array(hkl[i][p]).T).T, axis=1)
                      for p, Ai in enumerate(A[i])
                      ]

        # Calculate hkl direction of the sample normal
        hkl_grain[i] = [normalize(np.linalg.inv(Ai) @ nhat, axis=0) for Ai in A[i]]

        # xy locations for each of the specified hkl's
        xy[i] = np.sqrt(2)*np.array(np.abs(hkl_grain[i]))[:,[1,0],0]


    # Get standard lattice
    stdLattice = _calcStdLattice(0.648, 0.648, 0.648, 90, 90, 90)

    # Loop through the first pattern's reciprocal matrix for symmetry reduction
    angle = [None] * npos
    angle2 = [None] * npos
    rot = [None] * npos
    RX, RY, RZ = np.nan*np.zeros((3, npos))
    totalAngles = [None] * npos

    maxAngle = np.inf

    # Load symmetry operations
    symOps_fld = f'{os.path.dirname(os.path.abspath(__file__))}/symOps/'
    symOps = np.load(f'{symOps_fld}/SymmetryOps225.npz')['wData']

    for i in range(npos):
        if A[i] is None:
            continue

        gmi = A[i][0]
        angle[i], rot[i] = _symReducedRecipLattice(stdLattice, gmi, symOps)
        angle2[i], vec3 = _axisOfMatrix(rot[i], squareUp=1)
        vec3 *= np.tan(np.radians(angle2[i])/2)

        if angle2[i] <= maxAngle:
            RX[i], RY[i], RZ[i] = vec3.flatten()
            totalAngles[i] = angle2[i]

    # Rotation about the X-axis by +45 degrees
    rotFrame = R.from_rotvec([np.pi/4, 0, 0]).as_matrix()

    # Calculate rotations about H/F axes
    Rxhf = (np.linalg.inv(rotFrame) @ np.vstack((RX, RY, RZ))).T
    RH, RF = zip(*Rxhf[:, 1:])
    rotRGB = _calcRGB_JZT(RX, RH, RF, satAngle=None)

    # Collect sample info
    sampleInfo = {
                  'sampleX': sample_x,
                  'sampleY': sample_y,
                  'sampleZ': sample_z,
                  'sampleH': sample_h,
                  'sampleF': sample_f,
                  'eulerXHF': (RX, RH, RF),
                  }

    return rotRGB, sampleInfo


if __name__ == '__main__':
    prj_fld = '/mnt/c/Users/naveed/Documents/GitHub/xsd_mic/experiments/2023c1/34IDE/Laue/'
    xml_fp = glob(f'{prj_fld}/recon2D*.xml')
    rotRGB, sampleInfo = process(xml_fp[0])

    im = np.flip(rotRGB.reshape((81, 81, -1)).clip(min=0, max=1), axis=(0,1))
    sampleX = np.flip(sampleInfo['sampleX'].reshape((81,81)), axis=(0,1))
    sampleH = np.flip(sampleInfo['sampleH'].reshape((81,81)), axis=(0,1))
    minX, maxX = sampleX[0,0], sampleX[0,-1]
    minH, maxH = sampleH[0,0], sampleH[-1,0]

    plt.imshow(im, origin='lower', extent=(minX, maxX, minH, maxH))
    plt.show()
