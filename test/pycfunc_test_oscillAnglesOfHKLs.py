from timeit import default_timer as timer
import sys, os, math
import numpy as np

import numba.cuda

from hexrd.xrd import transforms as xf
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import pycfuncs_transforms as pycfuncs

bVec_ref    = np.ascontiguousarray(xf.bVec_ref)
eta_ref     = np.ascontiguousarray(xf.eta_ref)

idxFile = './ruby_4537-8_log.txt'

MAX_MULT_PYTHON = 24

#idxFile = './ruby_triple.txt'
gtable  = np.loadtxt(idxFile, delimiter='\t')
idx     = gtable[:, 0] >= 0
hklsT   = np.ascontiguousarray(gtable[idx, 2:5].T)
hkls    = np.ascontiguousarray(gtable[idx, 2:5])

# input parameters
wavelength = 0.153588                     # Angstroms (80.725keV)

chi    = -0.0011591608938627839

bMat = np.array( [ [  2.10048731e-01,   0.00000000e+00,   0.00000000e+00],
                   [  1.21271692e-01,   2.42543383e-01,   0.00000000e+00],
                   [  0.00000000e+00,   0.00000000e+00,   7.69486476e-02] ] )

rMat_c = xf.makeRotMatOfExpMap(np.array( [ [ 0.66931818],
                                           [-0.98578066],
                                           [ 0.73593251] ] ) )

# ######################################################################

def timed_run(n, fn, *args, **kwargs):
    t = timer()
    for i in range(n):
        res = fn(*args, **kwargs)
    t = timer() - t
    return (res, t)


def check_results(got, expected):
    return (np.allclose(got[0], expected[0]) and
            np.allclose(got[1], expected[1]))


def to_human_size_string(nbytes):
    unit = int(math.floor(math.log(nbytes, 1024)))
    unit_list = [ 'bytes', 'KiB', 'MiB', 'GiB', 'TiB' ]
    unit = min(unit, len(unit_list))
    return '{0:.2f} {1}'.format(nbytes/math.pow(1024, unit), unit_list[unit])


def run_test(N, array_mult=1):
    hkls_ = np.concatenate([hkls]*array_mult)

    if array_mult <= MAX_MULT_PYTHON:
        hklsT_ = np.concatenate([hklsT]*array_mult, axis=1)
        res_ref, t_ref = timed_run(N, xf.oscillAnglesOfHKLs, hklsT_, chi, rMat_c,
                                   bMat, wavelength, beamVec=bVec_ref,
                                   etaVec=eta_ref)
    else:
        res_ref, t_ref = None, None

    res_capi, t_capi = timed_run(N, xfcapi.oscillAnglesOfHKLs, hkls_, chi, rMat_c,
                                 bMat, wavelength, beamVec=bVec_ref,
                                 etaVec=eta_ref)

    gVec_e = np.zeros(3)
    gHat_c = np.zeros(3)
    gHat_s = np.zeros(3)
    bHat_l = np.zeros(3)
    eHat_l = np.zeros(3) 
    oVec = np.zeros(2)
    tVec0 = np.zeros(3)
    rMat_e = np.zeros(9)
    rMat_s = np.zeros(9)
    npts = hkls_.shape[0]
    #return arrays
    oangs0 = np.zeros((npts, 3))
    oangs1 = np.zeros((npts, 3))
    _, t_cuda = timed_run(N, pycfuncs.oscillAnglesOfHKLs, hkls_, chi, rMat_c, bMat,
                          wavelength, bVec_ref, eta_ref, gVec_e, gHat_c, gHat_s,
                          bHat_l, eHat_l, oVec, tVec0, rMat_e, rMat_s, npts,
                          oangs0, oangs1)
    res_cuda = (oangs0, oangs1)

    if res_ref:
        res_ref = (res_ref[0].T, res_ref[1].T)
        assert check_results(res_capi, res_ref)
    assert check_results(res_cuda, res_capi)

    return t_ref, t_capi, t_cuda


if __name__ == '__main__':
    import sys, getopt

    try:
        _, args = getopt.getopt(sys.argv[1:], 'f:')
    except:
        print('{0} -f <file> <list of multipliers for the file>')
        sys.exit(2)

    if not args:
        args = [ 1, 3, 12, 24, 96, 192, 384, 768, 1536, 3072, 6144 ]

    
    cuda_str = 'CUDA({0})'.format(numba.cuda.get_current_device().name)
    headers = ['SIZE', 'HEXRD', 'CAPI', cuda_str]
    print(', '.join('{0:>10}'.format(x) for x in headers))

    for i in args:
        try:
            sz = int(i)
        except ValueError:
            sz = 0

        if sz <= 0:
            print('Ignoring argument {0}: not a valid size'.format(i))
            continue

        res = run_test(22, array_mult=sz)
        sz_in_bytes = sz * hkls.nbytes
        res_str = ', '.join(['{:>10s}'.format(to_human_size_string(sz_in_bytes))] + 
                            ['{:>10s}'.format('not ran') 
                             if x is None else '{:10.6f}'.format(x)
                             for x in res])
        print(res_str)
