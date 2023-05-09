import numpy as np

from dask import array as da
from numpy.linalg import norm
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.sparse import linalg


def fwhm_nm2eV(x0_nm, sigma_nm):
    fwhm_nm = sigma2fwhm(sigma_nm)
    x0_eV = nm2eV(x0_nm)
    ub_nm = x0_nm + 0.5*fwhm_nm
    lb_nm = x0_nm - 0.5*fwhm_nm
    ub_eV = nm2eV(ub_nm)
    lb_eV = nm2eV(lb_nm)
    fwhm_eV = np.abs(ub_eV - lb_eV)
    sigma_eV = fwhm2sigma(fwhm_eV)

    return x0_eV, fwhm_eV, sigma_eV


def nm2eV(nm):
    return 1239.8/nm


def sigma2fwhm(sigma):
    return 2*np.sqrt(np.log(2))*sigma


def fwhm2sigma(fwhm):
    return fwhm/(np.sqrt(np.log(2)))


def gauss(x, x1, y1, sigma1):
    """
    Fits a 1D Gaussian curve to the input dataset.

    @author: Naveed Rahman <naveed@anl.gov>
    @references:

    Parameters
    ----------
    x : numpy.ndarray
        Input x values for calculating best f(x) values.
    x1 : float
        Center of Gaussian curve.
    y1 : float
        Amplitude of Gaussian curve.
    sigma1 : float
        Width of the Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma1.

    Returns
    -------
    numpy.ndarray
        Fitted f(x) values using the Gaussian parameters.
    """
    p = [x1, y1, sigma1]

    return p[1] * np.exp(-((x-p[0])/p[2])**2)


def df(x, x0, y0, sigma):
    """
    Calculates the Jacobian for a 1D Gaussian curve.

    @author: Naveed Rahman <naveed@anl.gov>
    @credits:   - Wolfram|Alpha used for calculating below derivatives.
                - See scipy.optimize.least_squares documentation for Jacobian.

    Parameters
    ----------
    x : numpy.ndarray
        Input x values for calculating each Jacobian.
    x0 : float
        Center of Gaussian curve.
    y0 : float
        Amplitude of Gaussian curve.
    sigma : float
        Width of the Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma.

    Returns
    -------
    numpy.ndarray
        Jacobians returned as an array: df/dx0, df/dy0, df/dsigma.
    """
    p = [x0, y0, sigma]

    dfx0 = (2*p[1] * (x-p[0]) * np.exp(-((x-p[0])/p[2])**2)) / p[2]**2
    dfy0 = np.exp(-((x-p[0])/p[2])**2)
    dfsigma = (2*p[1] * (x-p[0])**2 * np.exp(-((x-p[0])/p[2])**2)) / p[2]**3

    return np.transpose(np.array([dfx0, dfy0, dfsigma]))


def gauss_p0(ydata, xdata=None, filtWin=None, xcrop=None):
    if xdata is None:
        xdata = np.arange(0, len(ydata))

    # Smooth ydata for better estimates
    if filtWin is None:
        filtWin = len(ydata) // 24
    ydata = savgol_filter(ydata, filtWin, 3)

    # Peak center estimate
    if xcrop is not None:
        ind = np.argmax(ydata[xcrop:-xcrop]) + xcrop
        x0 = xdata[ind]
    else:
        x0 = xdata[np.argmax(ydata)]

    # Peak intensity estimate
    y0 = np.max(ydata)

    # Peak sigma estimate
    deriv = np.diff(ydata, n=1)
    fwhm = abs(xdata[deriv.argmin()] - xdata[deriv.argmax()])
    sigma = fwhm2sigma(fwhm)

    return (x0, y0, sigma)


def gauss_p0_stack(ydata, xdata, filtWin=None, xcrop=None):
    # Smooth ydata for better estimates
    if filtWin is None:
        filtWin = ydata.shape[1] // 24
    ydata = savgol_filter(ydata, filtWin, 3, axis=1)

    if xcrop is not None:
        ind = ydata[:,xcrop:-xcrop].argmax(axis=1) + xcrop
        x0 = np.take(xdata, ind)
    else:
        ind = ydata.argmax(axis=1)
        x0 = np.take(xdata, ind)

    # Peak intensity estimates
    y0 = ydata.max(axis=1)

    # Peak sigma estimates
    deriv = np.diff(ydata, axis=1, n=1)
    lb = np.take(xdata, deriv.argmax(axis=1))
    ub = np.take(xdata, deriv.argmin(axis=1))
    fwhm = np.abs(ub-lb)
    sigma = fwhm2sigma(fwhm)

    return (x0, y0, sigma)


def gauss2(x, x1, y1, sigma1, x2, y2, sigma2):
    """
    Fits two 1D Gaussian curves to the input dataset.

    @author: Naveed Rahman <naveed@anl.gov>
    @references:

    Parameters
    ----------
    x : numpy.ndarray
        Input x values for calculating best f(x) values.
    x1 : float
        Center of the primary Gaussian curve.
    y1 : float
        Amplitude of the primary Gaussian curve.
    sigma1 : float
        Width of the primary Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma.
    x2 : float
        Center of the secondary Gaussian curve.
    y2 : float
        Amplitude of the secondary Gaussian curve.
    sigma2 : float
        Width of the secondary Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma

    Returns
    -------
    numpy.ndarray
        Fitted f(x) values using the Gaussian parameters.
    """
    p = [x1, y1, sigma1, x2, y2, sigma2]

    f1 = p[1] * np.exp(-((x-p[0])/p[2])**2)
    f2 = p[4] * np.exp(-((x-p[3])/p[5])**2)
    fx = f1 + f2

    return fx


def df2(x, x1, y1, sigma1, x2, y2, sigma2):
    """
    Calculates the Jacobian for two 1D Gaussian curves.

    @author: Naveed Rahman <naveed@anl.gov>
    @credits:   - Wolfram|Alpha used for calculating below derivatives.
                - See scipy.optimize.least_squares documentation for Jacobian.

    Parameters
    ----------
    x : numpy.ndarray
        Input x values for calculating each Jacobian.
    x1 : float
        Center of the primary Gaussian curve.
    y1 : float
        Amplitude of the primary Gaussian curve.
    sigma1 : float
        Width of the primary Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma.
    x2 : float
        Center of the secondary Gaussian curve.
    y2 : float
        Amplitude of the secondary Gaussian curve.
    sigma2 : float
        Width of the secondary Gaussian curve. FWHM = 2*sqrt(ln 2)*sigma.

    Returns
    -------
    numpy.ndarray
        Jacobians returned as an array:
            df/dx1, df/dy1, df/dsigma1, df/dx2, df/dy2, df/dsigma2.
    """
    p = [x1, y1, sigma1, x2, y2, sigma2]

    dfx1 = (2*p[1] * (x-p[0]) * np.exp(-((x-p[0])/p[2])**2)) / p[2]**2
    dfy1 = np.exp(-((x-p[0])/p[2])**2)
    dfsigma1 = (2*p[1] * (x-p[0])**2 * np.exp(-((x-p[0])/p[2])**2)) / p[2]**3

    dfx2 = (2*p[4] * (x-p[3]) * np.exp(-((x-p[3])/p[5])**2)) / p[5]**2
    dfy2 = np.exp(-((x-p[3])/p[5])**2)
    dfsigma2 = (2*p[4] * (x-p[3])**2 * np.exp(-((x-p[3])/p[5])**2)) / p[5]**3

    return np.transpose(np.array([dfx1, dfy1, dfsigma1, dfx2, dfy2, dfsigma2]))


def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    """
    Removes baseline profile using a modified asymmetric least squares
    smoothing algorithm. Taken from: https://stackoverflow.com/a/67509948

    @authors: Rustam Guliev and Daniel Casas-Orozco

    Parameters
    ----------
    y : numpy.ndarray
        Singular spectrum to be corrected.
    ratio : float, optional
        Asymmetry parameter, 0.001 <= ratio <= 0.1 is good for positive peaks.
    lam : int, optional
        Smoothness parameter, 10E2 <= lam <= 10E9 generally good.
    niter : int, optional
        Number of iterations.
    full_output : bool, optional
        Boolean indicating whether or not to use return verbose output.

    Returns
    -------
    z : full_output=False
        Baseline spectrum.

    tuple(z, d, info) : full_output=True
        Baseline spectrum, baseline corrected values, and fitting information.
    """
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)

    # Transpose are flipped w.r.t. Algorithm on p. 25
    H = lam * D.dot(D.T)

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        # Clip the calculation to handle overflow errors
        exp64 = 709.78
        with np.errstate(over='ignore'):
            w_new = np.clip(1 / (1+np.exp(2*(d-(2*s-m))/s)), -exp64, exp64)

        crit = norm(w_new - w) / norm(w)

        w = w_new
        # Do not create a new matrix, just update diagonal values
        W.setdiag(w)

        count += 1

        if count > niter:
            print('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return z, d, info
    else:
        return z
