from timeit import default_timer as timer
import sys, os, math
import numpy as np

import numba.cuda
from numbapro import nvtx
from hexrd.xrd import transforms as xf
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import pycfuncs_transforms as pycfuncs

bVec_ref    = np.ascontiguousarray(xf.bVec_ref)
eta_ref     = np.ascontiguousarray(xf.eta_ref)

idxFile = './ruby_4537-8_log.txt'


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

def timed_run(N, fn, name, color):
    def _runner(*args, **kwargs):
        t = timer()
        for i in range(N):
            with nvtx.profile_range(name, color=color):
                res = fn(*args, **kwargs)
        t = timer()-t
        return res, t

    return _runner


def check_results(got, expected):
    return (np.allclose(got[0], expected[0]) and
            np.allclose(got[1], expected[1]))


def to_human_size_string(nbytes):
    unit = int(math.floor(math.log(nbytes, 1024)))
    unit_list = [ 'bytes', 'KiB', 'MiB', 'GiB', 'TiB' ]
    unit = min(unit, len(unit_list))
    return '{0:.2f} {1}'.format(nbytes/math.pow(1024, unit), unit_list[unit])


def run_test(experiments, N, array_mult=1):
    hkls_ = np.concatenate([hkls]*array_mult)

    if 'python' in experiments:
        hklsT_ = np.concatenate([hklsT]*array_mult, axis=1)
        xf_oscillAnglesOfHKLs = timed_run(N, xf.oscillAnglesOfHKLs,
                                          'python', nvtx.colors.blue)
        res_ref, t_ref = xf_oscillAnglesOfHKLs(hklsT_, chi, rMat_c, bMat,
                                               wavelength, beamVec=bVec_ref,
                                               etaVec=eta_ref)
        res_ref = (res_ref[0].T, res_ref[1].T)
    else:
        res_ref, t_ref = None, None


    if 'capi' in experiments:
        xfcapi_oscillAnglesOfHKLs = timed_run(N, xfcapi.oscillAnglesOfHKLs,
                                              'capi', color=nvtx.colors.green)
        res_capi, t_capi = xfcapi_oscillAnglesOfHKLs(hkls_, chi, rMat_c, bMat,
                                                     wavelength, beamVec=bVec_ref,
                                                     etaVec=eta_ref)
    else:
        res_capi, t_capi = None, None

    if 'cuda' in experiments:
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
        pycfuncs_oscillAnglesOfHKLs = timed_run(N, pycfuncs.oscillAnglesOfHKLs,
                                                'cuda numba', nvtx.colors.red)
        _, t_cuda = pycfuncs_oscillAnglesOfHKLs(hkls_, chi, rMat_c, bMat,
                                                wavelength, bVec_ref, eta_ref, 
                                                gVec_e, gHat_c, gHat_s, bHat_l,
                                                eHat_l, oVec, tVec0, rMat_e,
                                                rMat_s, npts, oangs0, oangs1)
        res_cuda = (oangs0, oangs1)
    else:
        res_cuda, t_cuda = None, None

    if res_ref is not None:
        if res_capi is not None:
            assert check_results(res_ref, res_capi)
        if res_cuda is not None:
            assert check_results(res_ref, res_cuda)

    return t_ref, t_capi, t_cuda


if __name__ == '__main__':
    import sys, getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'pac')
    except:
        print('{0} [-p] [-a] [-c] <list of multipliers for the file>')
        sys.exit(2)

    translate_opt = { '-p':'python', '-a':'capi', '-c':'cuda' }
    experiments = set(translate_opt[o] for o, _ in opts)

    if not args:
        args = [ 1, 3, 12, 24, 96, 192, 384, 768, 1536, 3072, 6144 ]

    headers = ['size']
    headers.extend(experiments)
    print(', '.join('{0:>10}'.format(x) for x in headers))

    for i in args:
        try:
            sz = int(i)
        except ValueError:
            sz = 0

        if sz <= 0:
            print('Ignoring argument {0}: not a valid size'.format(i))
            continue

        res = run_test(experiments, 22, array_mult=sz)
        sz_in_bytes = sz * hkls.nbytes
        res_str = ', '.join(['{:>10s}'.format(to_human_size_string(sz_in_bytes))] + 
                            ['{:>10s}'.format('not ran') 
                             if x is None else '{:10.6f}'.format(x)
                             for x in res])
        print(res_str)
