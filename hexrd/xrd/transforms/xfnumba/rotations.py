
import numba




@numba.guvectorize(['void(float64[:], float64[:,:])'],
                   '(m) -> (n,n)')
def rotMatOfQuat(Q4, M33):
    qw, qx, qy, qz = Q4
    M33[0,0] = 1 - 2*qy*qy - 2*qz*qz
    M33[0,1] = 2*qx*qy - 2*qz*qw
    M33[0,2] = 2*qx*qz - 2*qy*qw
    M33[1,0] = 2*qx*qy + 2*qz*qw
    M33[1,1] = 1 - 2*qx*qx - 2*qz*qz
    M33[1,2] = 2*qy*qz - 2*qx*qw
    M33[2,0] = 2*qx*qz - 2*qy*qw
    M33[2,1] = 2*qy*qz + 2*qx*qw
    M33[2,2] = 1 - 2*qx*qx - 2*qy*qy
