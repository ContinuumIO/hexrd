from timeit import default_timer as timer
import sys, os
#from numbapro import vectorize, float64, jit, guvectorize, autojit
import numpy as np
#from hexrd.xrd import nbdistortion as dFuncs
from hexrd.xrd import transforms as xf
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import pycfuncs_transforms as pycfuncs
import numba.cuda
from numbapro import nvtx

#input parameters
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

def timed_run(fn, name, color):
    def _runner(*args, **kwargs):
        t = timer()
        with nvtx.profile_range(name, color=color):
            res = fn(*args,**kwargs)
        t = timer()-t
        return (res, t)
    return _runner


def run_test(N, experiments):
    # ######################################################################
    # Calculate pixel coordinates

    pvec  = 204.8 * np.linspace(-1, 1, N)

    dcrds = np.meshgrid(pvec, pvec)
    XY    = np.vstack([dcrds[0].flatten(), dcrds[1].flatten()]).T

    dangs = xf.detectorXYToGvec(XY, rMat_d, rMat_s, tVec_d, tVec_s, tVec_c,
                              beamVec=bVec_ref)
    ((tTh_d1, eta_d1), gVec_l1) = dangs

    dangs2 = xfcapi.detectorXYToGvec(XY, rMat_d, rMat_s, tVec_d, tVec_s, tVec_c,
                                     beamVec=bVec_ref)
    ((tTh_d2, eta_d2), gVec_l2) = dangs2

    gVec_c1 = np.dot(rMat_c.T, np.dot(rMat_s.T,gVec_l1))
    gVec_c2 = np.ascontiguousarray(np.dot(rMat_c.T, np.dot(rMat_s.T, gVec_l2.T)).T)

    time_deltas = []

    if 'python' in experiments:
        xf_gvecToDetector = timed_run(xf.gvecToDetectorXY,
                                      'python', nvtx.colors.blue)
        res_ref, t = xf_gvecToDetectorXY(gVec_c1, rMat_d, rMat_s, rMat_c,
                                         tVec_d, tVec_s, tVec_c,
                                         beamVec=bVec_ref)
        time_deltas.append(t)
    else:
        res_ref = None

    if 'capi' in experiments:
        xfcapi_gvecToDetectorXY = timed_run(xfcapi.gvecToDetectorXY,
                                            'capi', nvtx.colors.green)
        res_capi, t = xfcapi_gvecToDetectorXY(gVec_c2, rMat_d, rMat_s, rMat_c,
                                              tVec_d, tVec_s, tVec_c,
                                              beamVec=bVec_ref)
        time_deltas.append(t)
    else:
        res_capi = None

    # setup or numba version
    # should be able to run in nopython mode
    if 'cuda' in experiments:
        bHat_l = np.zeros(3)
        nVec_l = np.zeros(3)
        P0_l = np.zeros(3)
        P2_l = np.zeros(3)
        P2_d = np.zeros(3)
        P3_l = np.zeros(3)
        gHat_c = np.zeros(3)
        gVec_l = np.zeros(3)
        dVec_l = np.zeros(3)
        rMat_sc = np.zeros(9)
        brMat = np.zeros(9)
        result = np.empty((gVec_c2.shape[0], 3))
        bVec_ref_flat = bVec_ref.flatten()

        pycfuncs_gvecToDetectorXY = timed_run(pycfuncs.gvecToDetectorXY,
                                              'cuda numba', nvtx.colors.red)
        _, t = pycfuncs_gvecToDetectorXY(gVec_c2, rMat_d, rMat_s, rMat_c, 
                                         tVec_d, tVec_s, tVec_c,
                                         bVec_ref_flat, bHat_l, nVec_l, P0_l,
                                         P2_l, P2_d, P3_l, gHat_c, gVec_l,
                                         dVec_l, rMat_sc, brMat, result)
        res_cuda = result[:, 0:2]
        time_deltas.append(t)
    else:
        res_cuda = None

    if res_ref is not None:
        if res_capi is not None:
            assert np.allclose(res_ref, res_capi)
        if res_cuda is not None:
            assert np.allclose(res_ref, res_cuda)

    return time_deltas


if __name__ == '__main__':
    import sys, getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'pac')

    except getopt.GetoptError:
        print('{0} <list of sizes>'.format(sys.argv[0]))
        sys.exit(2)

    translate_opt = { '-p':'python', '-a':'capi', '-c':'cuda' }
    experiments = set(translate_opt[o] for o, _ in opts)

    if not args:
        args = [ 512, 1024, 2048, 4096 ]

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

        res = run_test(sz, experiments)
        res_str = ', '.join(['{:10d}'.format(sz)] + ['{:10.6f}'.format(x) for x in res])
        print(res_str)
