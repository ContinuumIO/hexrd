from __future__ import division, print_function
from math import isnan, fabs, cos, sin, acos, pi, atan2, asin, sqrt
from numpy import hypot
import numpy as np
import numba as nb
hypot2 = hypot

TARGET = 'cpu'
if TARGET == 'cuda':
    from numba import cuda
    def jit(signature):
        return cuda.jit(signature, device=True)
    def guvectorize(sig1, sig2):
        return nb.guvectorize(sig1, sig2, nopython=True, target='cuda')
    @jit('f8(f8,f8)')
    def hypot2(x,y):
        return sqrt(x**2+y**2)

elif TARGET == 'parallel':
    def jit(signature):
        return nb.jit(signature, nopython=True, nogil=True, target='parallel')
    def guvectorize(sig1, sig2):
        return nb.guvectorize(sig1, sig2, nopython=True, target='parallel')

else:
    def jit(signature):
        return nb.jit(signature, nopython=True, nogil=True)
    def guvectorize(sig1, sig2):
        return nb.guvectorize(sig1, sig2, nopython=True)


@jit('f8(f8[:])')
def norm3_(vec):
    return hypot2(hypot2(vec[0], vec[1]), vec[2])

@guvectorize('(f8[:],f8[:])','(n)->()')
def norm3_gufunc_(vec, out):
    out[0] = norm3_(vec)

def norm3(vec, out=None):
    '''
    Optimized vector norm for 3 vectors (and arrays of same).
    '''
    if vec.shape[-1] != 3:
        raise RuntimeError('Array of 3-vectors expected')
    return norm3_gufunc_(vec, out)


@jit('(f8[:],f8,f8[:])')
def div3_(vec, scl, out):
    out[0] = vec[0] / scl
    out[1] = vec[1] / scl
    out[2] = vec[2] / scl

@guvectorize('(f8[:],f8,f8[:])','(n),()->(n)')
def div3_gufunc_(vec, scl, out):
	div3_(vec, scl, out)

def div3(vec, scl, out=None):
    '''
    Optimized division for 3 vectors (and arrays of same).
    '''
    if vec.shape[-1] != 3:
        raise RuntimeError('Array of 3-vectors expected')
    return div3_gufunc_(vec, scl, out)
    

@jit('f8(f8[:],f8[:])')
def normalize3_(vec, out):
    tmp = norm3_(vec)
    div3_(vec, tmp, out)
    return tmp

@guvectorize('(f8[:],f8[:])','(n)->()')
def normalize3_gufunc_(vec, out):
    out[0] = normalize3_(vec, vec)

def normalize3(vec, out=None):
    '''
    Normalizing 3 vectors (and arrays of same).
    '''
    if vec.shape[-1] != 3:
        raise RuntimeError('Array of 3-vectors expected')
    return normalize3_gufunc_(vec, out)


@jit('f8(f8[:],f8[:])')
def dot3_(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@guvectorize('(f8[:],f8[:],f8[:])', '(n),(n)->()')
def dot3_gufunc_(a, b, out):
    out[0] = dot3_(a, b)

def dot3(a, b):
    '''
    Optimized dot product for 3 vectors (and arrays of same).
    '''
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise RuntimeError('Inputs must be arrays of 3-vectors')
    return dot3_gufunc_(a, b)


@jit('(f8[:],f8[:,:])')
# Reference: http://fabiensanglard.net/doom3_documentation/37726-293748.pdf
# I needed to swap elements and negate some signs to get the results to
# match up with the original C code. I'm assuming this is due to differences
# in the coordinate systems assumed. But it avoids the use of trigonometric
# functions, which is certainly a win.
def quatToRotMat_(q, m):
    x2 = 2.0 * q[0]
    y2 = 2.0 * q[1]
    z2 = 2.0 * q[2]

    xx2 = q[0] * x2
    yy2 = q[1] * y2
    zz2 = q[2] * z2
    m[0,0] = ( xx2 + yy2 ) - 1.0
    m[1,1] = ( xx2 + zz2 ) - 1.0
    m[2,2] = 1.0 - ( yy2 + zz2 )

    yz2 = q[1] * z2
    wx2 = q[3] * x2
    m[0,1] = yz2 - wx2
    m[1,0] = yz2 + wx2

    xy2 = q[0] * y2
    wz2 = q[3] * z2
    m[1,2] = wz2 - xy2
    m[2,1] = wz2 + xy2

    xz2 = q[0] * z2
    wy2 = q[3] * y2
    m[2,0] = wy2 - xz2
    m[0,2] = wy2 + xz2

@guvectorize('(f8[:],f8[:],f8[:,:])', '(m),(n)->(n,n)')
def quatToRotMat_gufunc_(q, d3, m):
    quatToRotMat_(q, m)

def quatToRotMat(quats):
    """
    Make rotation matrices from unit quaternions

    """
    if quats.shape[-1] != 4:
        raise RuntimeError('Input must be an array of quaternions (4-vectors)')
    return quatToRotMat_gufunc_(quats, np.empty((3,)))


sqr05 = sqrt(0.5)
@nb.jit('f8[:,:](f8[:])', nopython=True)
def vecMVToSymm(A):
    symm = np.empty((3,3))
    symm[0, 0] = A[0]
    symm[0, 1] = symm[1, 0] = A[5] * sqr05
    symm[0, 2] = symm[2, 0] = A[4] * sqr05
    symm[1, 1] = A[1]
    symm[1, 2] = symm[2, 1] = A[3] * sqr05
    symm[2, 2] = A[2]
    return symm


etaFrameTol = 1.0 - sqrt(np.finfo(np.float32).eps)
@nb.jit('f8[:,:](f8[:],f8[:])', nopython=True)
def etaFrameToRotMat(beamVec, etaVec):
    rMat = np.empty((3,3))
    tmp = hypot(hypot(beamVec[0], beamVec[1]), beamVec[2])
    rMat[:, 2] = beamVec / -tmp
    dt = rMat[0, 2] * etaVec[0] + rMat[1, 2] * etaVec[1] + rMat[2, 2] * etaVec[2]
    tmp = hypot(hypot(etaVec[0], etaVec[1]), etaVec[2])
    if fabs(dt) >= tmp * etaFrameTol:
        rMat[0, 0] = rMat[1, 0] = rMat[2, 0] = 0.0
        rMat[0, 1] = rMat[1, 1] = rMat[2, 1] = np.nan
    else:
        rMat[0, 0] = etaVec[0] - dt * rMat[0, 2]
        rMat[1, 0] = etaVec[1] - dt * rMat[1, 2]
        rMat[2, 0] = etaVec[2] - dt * rMat[2, 2]
        tmp = hypot(hypot(rMat[0, 0], rMat[1, 0]), rMat[2, 0])
        rMat[:, 0] /= tmp
        rMat[0, 1] = rMat[1, 2] * rMat[2, 0] - rMat[2, 2] * rMat[1, 0]
        rMat[1, 1] = rMat[2, 2] * rMat[0, 0] - rMat[0, 2] * rMat[2, 0]
        rMat[2, 1] = rMat[0, 2] * rMat[1, 0] - rMat[1, 2] * rMat[0, 0]
    return rMat


@jit('(f8[:,:],f8[:],f8[:])')
def mv3_(a, b, c):
    c[0] = dot3_(a[0], b)
    c[1] = dot3_(a[1], b)
    c[2] = dot3_(a[2], b)

@jit('(f8[:],f8,f8,f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def oscillAnglesOfHKL_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, scratch, oangs):
    # reciprocal lattice vector in SAMPLE frame
    tmp = scratch[0]
    gHat_s = scratch[1]
    mv3_(bMat, hkl, gHat_s)
    mv3_(rMat_c, gHat_s, tmp)
    mv3_(vMat_s, tmp, gHat_s)

    # Normalize
    nrm0 = normalize3_(gHat_s, gHat_s)
    # sin of the Bragg angle assoc. with wavelength
    sintht = 0.5 * wavelength * nrm0              

    # coefficients for harmonic equation
    # The normalized beam vector is the negative of the third column of
    # the eta frame rotation matrix.
    bHat = rMat_e[:,2]
    t1 = sc * bHat[1] - cc * bHat[2]
    t2 = cc * bHat[1] + sc * bHat[2]
    a = - gHat_s[2] * bHat[0] - gHat_s[0] * t1
    b = - gHat_s[0] * bHat[0] + gHat_s[2] * t1
    c =               -sintht + gHat_s[1] * t2

    # form solution
    abMag = hypot2(a,b)
    phaseAng = atan2(b,a)
    rhs = c / abMag

    # quick exit for an infeasible case
    if fabs(rhs) > 1.0:
        oangs[0,0] = oangs[0,1] = oangs[0,2] = \
            oangs[1,0] = oangs[1,1] = oangs[1,2] = np.nan
        return

    rhsAng = asin(rhs)
    oangs[0,2] = rhsAng - phaseAng
    oangs[1,2] = pi - rhsAng - phaseAng

    oangs[0,0] = oangs[1,0] = 2.0 * asin(sintht)
    for i in range(2):
        co = cos(oangs[i, 2])
        so = sin(oangs[i, 2])
        tmp[0] = co * gHat_s[0] + so * gHat_s[2]
        tmp[1] = cc * gHat_s[1] + sc * ( so * gHat_s[0] - co * gHat_s[2] )
        tmp[2] = sc * gHat_s[1] + cc * ( co * gHat_s[2] - so * gHat_s[0] )
        oangs[i,1] = atan2(dot3_(tmp, rMat_e[:,1]), dot3_(tmp, rMat_e[:,0]))

if TARGET=='cuda':
    def oscillAnglesOfHKL_gufunc_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, dummy2, oangs):
        scratch = cuda.local.array((2,3), dtype=nb.float64)
        oscillAnglesOfHKL_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, scratch, oangs)
else:
    def oscillAnglesOfHKL_gufunc_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, dummy2, oangs):
        scratch = np.empty((2,3))
        oscillAnglesOfHKL_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, scratch, oangs)
oscillAnglesOfHKL_gufunc_ = guvectorize(
        '(f8[:],f8,f8,f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:],f8[:],f8[:,:])',
        '(n3),(),(),(n3,n3),(n3,n3),(),(n3,n3),(n3,n4),(n2)->(n2,n3)'
        )(oscillAnglesOfHKL_gufunc_)

VEC2 = np.empty((2,))
vInv_ref = np.array([1.,1.,1.,0.,0.,0.])
bVec_ref = np.array([0.,0.,-1.])
eta_ref = np.array([1.,0.,0.])
def oscillAnglesOfHKL(hkl, chi, rMat, bMat, wavelength, vInv_s=vInv_ref, beamVec=bVec_ref, etaVec=eta_ref):
    vMat_s = vecMVToSymm(vInv_s)                # stretch tensor in SAMPLE frame
    rMat_e = etaFrameToRotMat(beamVec, etaVec)  # eta basis COB with beam antiparallel with Z
    nrmat = rMat.ndim - 2
    nhkl = hkl.ndim - 1
    if nrmat > 0:
        hkl = hkl.reshape((1,) * nrmat + hkl.shape)
    if nhkl > 0:
        rMat = rMat.reshape(rMat.shape[:1] + ((1,) * nhkl) + rMat.shape[1:])
    return oscillAnglesOfHKL_gufunc_(hkl, np.cos(chi), np.sin(chi), rMat, bMat, wavelength, 
        vMat_s, rMat_e, VEC2)


@jit('i8(f8,f8[:])')
def find_in_range_(value, spans):
    """find the index in spans where value >= spans[i] and value < spans[i].

    spans is an ordered array where spans[i] <= spans[i+1] (most often <
    will hold).

    If value is not in the range [spans[0], spans[-1][, then -2 is returned.

    This is equivalent to "bisect_right" in the bisect package, in which
    code it is based, and it is somewhat similar to NumPy's searchsorted,
    but non-vectorized

    """
    if value < spans[0] or value >= spans[-1]:
        return -2
    # from the previous check, we know 0 is not a possible result
    li = 0
    ri = len(spans)
    while li < ri:
        mi = (li + ri) // 2
        if value < spans[mi]:
            ri = mi
        else:
            li = mi+1
    return li


twopi = 2 * np.pi
@jit('(f8[:],f8,f8[:,:], f8,f8[:],f8[:],i8, f8,f8[:],f8[:],i8)')
def angle_is_hit_(angles, threshold, etaOmeMap,
                  eta_offset, valid_eta_spans, etaEdges, dpix_eta,
                  ome_offset, valid_ome_spans, omeEdges, dpix_ome):
    """perform work on one of the angles.

    This includes:
    - filtering nan values
    - filtering out angles not in the specified spans
    - checking that the discretized angle fits into the sensor range (maybe
      this could be merged with the previous test somehow, for extra speed)
    - actual check for a hit, using dilation for the tolerance.
    Note the function returns both, if it was a hit and if it passed the the
    filtering, as we'll want to discard the filtered values when computing
    the hit percentage.

    """
    if isnan(angles[0]):
        return (0,0)

    eta = ((angles[1] - eta_offset) % twopi) + eta_offset
    if find_in_range_(eta, valid_eta_spans) & 1 == 0:
        # index is even: out of valid eta spans
        return (0,0)

    ome = ((angles[2] - ome_offset) % twopi) + ome_offset
    if find_in_range_(ome, valid_ome_spans) & 1 == 0:
        # index is even: out of valid ome spans
        return (0,0)

    # discretize the angles
    eta_idx = find_in_range_(eta, etaEdges) - 1
    if eta_idx < 0:
        # out of range
        return (0,0)

    ome_idx = find_in_range_(ome, omeEdges) - 1
    if ome_idx < 0:
        # out of range
        return (0,0)

    ome_start = max(ome_idx - dpix_ome, 0)
    ome_stop  = min(ome_idx + dpix_ome + 1, etaOmeMap.shape[0])
    eta_start = max(eta_idx - dpix_eta, 0)
    eta_stop  = min(eta_idx + dpix_eta + 1, etaOmeMap.shape[1])
    for i in range(ome_start, ome_stop):
        for j in range(eta_start, eta_stop):
            if etaOmeMap[i,j] > threshold:
                return (1, 1)
                
    return (0, 1)


tiny64 = np.finfo(np.float64).tiny
@jit('(f8[:], f8[:,:],f8,f8,f8[:,:],f8,f8[:,:],f8[:,:], i8[:],f8[:],f8[:,:,:], f8,f8[:],f8[:],i8, f8,f8[:],f8[:],i8, f8[:,:])')
def paintGridThis_(quat,
    hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e, 
    hkl_ix, thresholds, etaOmeMaps,
    offset_eta, valid_eta_spans, etaEdges, dpix_eta,
    offset_ome, valid_ome_spans, omeEdges, dpix_ome, scratch):
    rMat = scratch[:3]
    oangs = scratch[3:5]
    scratch2 = scratch[5:]
    quatToRotMat_(quat, rMat)
    total = hits = 0
    for i in range(hkls.shape[0]):
        oscillAnglesOfHKL_(hkls[i], cc, sc, rMat, bMat, wavelength, vMat_s, rMat_e, scratch2, oangs)
        etaOmeMap = etaOmeMaps[hkl_ix[i]]
        threshold = thresholds[hkl_ix[i]]
        for j in range(2):
            oang = oangs[j]
            nh, nt = angle_is_hit_(oang, threshold, etaOmeMap,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome)
            hits += nh
            total += nt
    return float(hits) / (float(total) + tiny64)

if TARGET=='cuda':
    def paintGridThis_gufunc_(quat,
        hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e, 
        hkl_ix, threshold, etaOmeMaps,
        offset_eta, valid_eta_spans, etaEdges, dpix_eta,
        offset_ome, valid_ome_spans, omeEdges, dpix_ome, out):
        out[0] = paintGridThis_(quat,
                hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e,
                hkl_ix, threshold, etaOmeMaps,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome, 
                cuda.local.array((7,3), dtype=nb.float64))
else:
    def paintGridThis_gufunc_(quat,
        hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e, 
        hkl_ix, threshold, etaOmeMaps,
        offset_eta, valid_eta_spans, etaEdges, dpix_eta,
        offset_ome, valid_ome_spans, omeEdges, dpix_ome, out):
        out[0] = paintGridThis_(quat,
                hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e,
                hkl_ix, threshold, etaOmeMaps,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome,
                np.empty((7,3)))
paintGridThis_gufunc_ = guvectorize(
    '(f8[:], f8[:,:],f8,f8,f8[:,:],f8,f8[:,:],f8[:,:], i8[:],f8[:],f8[:,:,:],  f8,f8[:],f8[:],i8,f8,f8[:],f8[:],i8, f8[:])',
    '(n4),   (mm,n3),(),(),(n3,n3),(),(n3,n3),(n3,n3), (mm), (m1), (m1,m2,m3), (), (m4), (m5),(),(),(m7), (m8), () -> ()'
    )(paintGridThis_gufunc_)

def paintGridThis(quats,
    hkls, chi, bMat, wavelength, vInv_s, beamVec, etaVec,
    hkl_ix, threshold, etaOmeMaps,
    offset_eta, valid_eta_spans, etaEdges, dpix_eta,
    offset_ome, valid_ome_spans, omeEdges, dpix_ome):
    if quats.ndim == 0 or quats.shape[-1] != 4:
        raise RuntimeError('An array of quaternions is expected, but quats.shape[-1] != 4')
    vMat_s = vecMVToSymm(vInv_s)                # stretch tensor in SAMPLE frame
    rMat_e = etaFrameToRotMat(beamVec, etaVec)  # eta basis COB with beam antiparallel with Z
    cc = cos(chi)
    sc = sin(chi)
    return paintGridThis_gufunc_(quats,
                hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e,
                hkl_ix, threshold, etaOmeMaps,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome)

