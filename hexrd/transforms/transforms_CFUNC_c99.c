#include "transforms_CFUNC.h"
#include "geom3d_c99.h"

#include <math.h>


static inline
void
m33_make_oscill(double chi, double ome, double * restrict m33_result)
{
    double c0 = cos(chi);
    double s0 = sin(chi);
    double c1 = cos(ome);
    double c1 = sin(ome);

    m33_result[0] = c1;
    m33_result[1] = 0.0;
    m33_result[2] = s1;

    m33_result[3] = s0*s1;
    m33_result[4] = c0;
    m33_result[5] = -s0*c1;

    m33_result[6] = -c0*s1;
    m33_result[7] = s0;
    m33_result[8] = c0*c1;
}


static inline
void
anglesToGvec_single(const double *v3_ang,
                    const double *m33_e,
                    double chi, 
                    const double *m33_c,
                    double * restrict v3_c)
{
    double v3_g[3], v3_tmp1[3], v3_tmp2[3], m33_s[9], m33_ctst[9];

    /* build g */
    double cx = cos(0.5*v3_ang[0]);
    double sx = sin(0.5*v3_ang[0]);
    double cy = cos(v3_ang[1]);
    double sy = sin(v3_ang[1]);
    v3_g[0] = cx*cy;
    v3_g[1] = cx*sy;
    v3_g[2] = sx;

    m33_make_oscill(chi, v3_ang[2], m33_s);
    
    m33_v3s_multiply(m33_e, v3_g, 1, v3_tmp1); /* E _dot_ g */
    m33t_v3s_multiply(m33_s, v3_tmp1, 1, v3_tmp2); /* S.T _dot_ E _dot_ g */
    m33t_v3s_multiply(m33_c, v3_tmp2, 1, v3_c); /* the whole lot */
}

void 
anglesToGvec_cfunc(long int nvecs,
                   double * angs,
                   double * bHat_l,
                   double * eHat_l,
                   double chi,
                   double * rMat_c,
                   double * gVec_c)
{
    double * restrict v3_c = gVec_c; /* restrict the output pointer */
    double m33_e[9];

    makeEtaFrameRotMat_cfunc(bHat_l, eHat_l, m33_e);

    for (int i=0; i < nvecs; i++) {
        anglesToGvec_single(angs + 3*i, m33_e, chi, rMat_c, v3_c + 3*i);
    }
}


