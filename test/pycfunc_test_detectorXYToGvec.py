from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
import sys, os, time
import numpy as np

from hexrd.xrd import transforms as xf
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import pycfuncs_transforms as pycfuncs

import numba.cuda

# input parameters
bVec_ref = xf.bVec_ref

rMat_d = xf.makeDetectorRotMat( ( 0.0011546340766314521,
                                 -0.0040527538387122993,
                                 -0.0026221336905160211 ) )
tVec_d = np.array( [ [   -1.44904 ],
                     [   -3.235616],
                     [-1050.74026 ] ] )

chi    = -0.0011591608938627839
tVec_s = np.array([ [-0.15354144],
                    [ 0.        ],
                    [-0.23294777] ] )

rMat_c = xf.makeRotMatOfExpMap(np.array( [ [ 0.66931818],
                                           [-0.98578066],
                                           [ 0.73593251] ] ) )
tVec_c = np.array( [ [ 0.07547626],
                     [ 0.08827523],
                     [-0.02131205] ] )

rMat_s = xf.makeOscillRotMat([chi, 0.])


def timed_run(fn, *args, **kwargs):
    t = timer()
    res = fn(*args,**kwargs)
    t = timer()-t
    return (res, t)


def check_results(got, expected):
    return (np.allclose(got[0][0], expected[0][0]) and
            np.allclose(got[0][1], expected[0][1]) and
            np.allclose(got[1], expected[1]))


def run_test(N):
    # ##################################################################
    # Calculate pixel coordinates
    #
    pvec  = 204.8 * np.linspace(-1, 1, N)
    dcrds = np.meshgrid(pvec, pvec)
    XY    = np.ascontiguousarray(np.vstack([dcrds[0].flatten(),
                                            dcrds[1].flatten()]).T)

    # Check the timings
    res_ref, t_ref = timed_run(xf.detectorXYToGvec, XY, rMat_d, rMat_s,
                               tVec_d, tVec_s, tVec_c, beamVec=bVec_ref)

    res_ref = [res_ref[0], res_ref[1].T]

    res_capi, t_capi = timed_run(xfcapi.detectorXYToGvec, XY, rMat_d,
                                 rMat_s, tVec_d.flatten(),
                                 tVec_s.flatten(), tVec_c.flatten(),
                                 beamVec=bVec_ref.flatten(),
                                 etaVec=np.array([1.0, 0.0, 0.0]))

    #maxDiff_tTh = np.linalg.norm(tTh_d1-tTh_d3,np.inf)
    #print("Maximum disagreement in tTh:  %f"%maxDiff_tTh)
    #maxDiff_eta = np.linalg.norm(eta_d1-eta_d3,np.inf)
    #print("Maximum disagreement in eta:  %f"%maxDiff_eta)
    #maxDiff_gVec = np.linalg.norm(np.sqrt(np.sum(np.asarray(gVec1.T-gVec3)**2,1)),np.inf)
    #print("Maximum disagreement in gVec: %f"%maxDiff_gVec)

    #setup for detectorXYToGVec
    rMat_e = np.zeros(9)
    bVec = np.zeros(3)
    tVec1 = np.zeros(3)
    tVec2 = np.zeros(3)
    dHat_l = np.zeros(3)
    n_g = np.zeros(3)
    npts = XY.shape[0]
    #return values
    tTh = np.zeros(npts)
    eta = np.zeros(npts)
    gVec_l = np.zeros((npts, 3))

    _, t_cuda = timed_run(pycfuncs.detectorXYToGvec, XY, rMat_d, rMat_s,
                          tVec_d.flatten(), tVec_s.flatten(),
                          tVec_c.flatten(),
                          bVec_ref.flatten(),np.array([1.0,0.0,0.0]),
                          rMat_e, bVec, tVec1, tVec2, dHat_l, n_g, npts,
                          tTh, eta, gVec_l)

    res_cuda = [[tTh, eta], gVec_l]
    #maxDiff_tTh = np.linalg.norm(tTh_d3 - tTh_d4,np.inf)
    #print("Maximum disagreement in tTh:  %f"%maxDiff_tTh)
    #maxDiff_eta = np.linalg.norm(eta_d3 - eta_d4,np.inf)
    #print("Maximum disagreement in eta:  %f"%maxDiff_eta)
    #maxDiff_gVec = np.linalg.norm(np.sqrt(np.sum(np.asarray(gVec3 - gVec4)**2,1)),np.inf)
    #print("Maximum disagreement in gVec: %f"%maxDiff_gVec)

    assert check_results(res_capi, res_ref)
    assert check_results(res_cuda, res_ref)

    return t_ref, t_capi, t_cuda


if __name__ == '__main__':
    import sys, getopt
    try:
        _, args = getopt.getopt(sys.argv[1:], '')
    except getopt.GetoptError:
        print('{0} <list of sizes>'.format(sys.argv[0]))
        sys.exit(2)

    if not args:
        args = [ 512, 1024, 2048, 4096 ]


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

        res = run_test(sz)
        res_str = ', '.join(['{:10d}'.format(sz)] + ['{:10.6f}'.format(x) for x in res])
        print(res_str)
