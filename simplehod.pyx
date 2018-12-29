#cython: language_level=2, boundscheck=True
#cython: embedsignature=True

import numpy
from libc.math cimport log, sqrt, sin, cos, erf, exp
from libc.math cimport M_PI as PI
from numpy.random import RandomState

def mkhodp_linear(afof, apiv, hod_dMda, mcut_apiv, m1_apiv):
    """ Generate linearly evolving HOD parameter based
        on values at a pivot, a=apiv, for halos at scale factor afof.

        returns mcut, m1 th
    """
    dm = (afof - apiv) * hod_dMda
    return mcut_apiv * (1 + dm), m1_apiv * (1 + dm)

def hod(rng, mfof,
    pos=None, vel=None, conc=None, rvir=None, vdisp=None,
    mcut=1e13, sigma=0.2, m1=1e12, kappa=0.8,
    alpha=0.444, vcen=1, vsat=1
    ):
    """
    Apply HOD to a halo catalog.

    rng is a random seed or a RandomState object.
    Do not reuse the rng, as the number of draws from the rng
    may change as this code evolves

    pos, vel shall have the same length as mfof.
   
    Every other variable is broadcast against mfof.

    if pos is None, only return (ncen, nsat) integer number of objects
    for centrals and satellites.

    """ 
    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    seed1, seed2, seed3 = rng.randint(0x7fffffff, size=3)

    ncen, nsat = mkn(
            mfof=mfof,
            mcut=mcut, sigma=sigma, m1=m1, kappa=kappa,
            alpha=alpha)


    ncen, nsat = mknint(RandomState(seed1), ncen, nsat)

    if pos is None:
        return ncen, nsat
    else:
        cpos, cvel = mkcen(RandomState(seed2), ncen,
            pos, vel, vdisp, vcen=vcen)

        spos, svel = mksat(RandomState(seed3), nsat,
            pos, vel, vdisp, conc, rvir, vsat=vsat)

        return (ncen, cpos, cvel), (nsat, spos, svel)


def mknint(rng, ncen, nsat):
    """ generate integer samples of ncen and nsat.

        This step is needed after mkn if you want the exact
        number of objects.

        ncen is draw from a binomial with n = 1 (thus we
        at most have 1 ncen)

        nsat is draw from a poisson.
    """

    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    ncen = rng.binomial(1, ncen)
    nsat = rng.poisson(nsat)

    return ncen, nsat

def mkn(mfof,
    mcut=10**13.35, sigma=0.25, m1=10**12.8, kappa=1.0,
    alpha=0.8):
    """
    Main HOD model.

    rng is a random seed or a RandomState object.
    Do not reuse the rng, as the number of draws from the rng
    may change as this code evolves

    mcut is the cut off halo mass for centrals
    sigma is the scattering of mcut
    m1 is the mass per satellite
    kappa and alpha controls number of satellites.

    Returns ncen, nsat as expected number of galaxies per halo.

    use numpy.random.binomial to draw nsat from the expection.
    use numpy.random.poisson to draw nsat from the expection.

    """

    mfof, mcut, sigma, m1, kappa, alpha = numpy.broadcast_arrays(
            mfof, mcut, sigma, m1, kappa, alpha)

    ncen, nsat = _mkn(mfof=mfof.astype('=f4'),
            mcut=mcut.astype('=f4'),
            sigma=sigma.astype('=f4'),
            m1=m1.astype('=f4'),
            kappa=kappa.astype('=f4'),
            alpha=alpha.astype('=f4'))

    return ncen, nsat

cdef _mkn(
        const float [:] mfof,
        const float [:] mcut,
        const float [:] sigma,
        const float [:] m1,
        const float [:] kappa,
        const float [:] alpha,
):

    cdef int igrp

    cdef float [:] ncen
    cdef float [:] nsat

    ncen = numpy.zeros(mfof.shape[0], dtype='f4')
    nsat = numpy.zeros(mfof.shape[0], dtype='f4')

    for igrp in range(0, mfof.shape[0]):

        mass = mfof[igrp]
        logm = log(mass / mcut[igrp]) 

        ncen[igrp] = 0

        if sigma[igrp] <= 0:
            if mass > mcut[igrp] :
                ncen[igrp] = 1
        else:
            if logm > -5 * sigma[igrp]:
                mu = 0.5*(1+erf(logm/sqrt(2.0)/sigma[igrp]))
                ncen[igrp] = mu

        # sats for cen
        if mass > kappa[igrp]*mcut[igrp]:
            mu = ((mass-kappa[igrp]*mcut[igrp])/m1[igrp]) ** alpha[igrp]
            nsat[igrp] = mu

    return numpy.array(ncen), numpy.array(nsat)

def mkcen(rng, ncen,
    pos, vel, vdisp,
    vcen=0
    ):
    """
    Central galaxy HOD model

    rng is a random seed or a RandomState object.
    Do not reuse the rng, as the number of draws from the rng
    may change as this code evolves

    vdisp is the DM velocity dispersion of the halo.
    
    vcen is the fraction of velocity relative to the dispersion.

    Returns (cpos, cvel), position and velocity of centrals.

    """
    if not isinstance(rng, RandomState):
        rng = RandomState(rng)
    vcen = numpy.broadcast_to(vcen, ncen.shape)
    vdisp = numpy.broadcast_to(vdisp, ncen.shape)
    
    rnga = RNGAdapter(rng, min(len(ncen), 1024 * 8))

    return _mkcen(rnga,
                ncen=ncen.astype('=i4'),
                pos=pos.astype('=f4'),
                vel=vel.astype('=f4'),
                vdisp=vdisp.astype('=f4'),
                vcen=vcen.astype('=f4'))

cdef _mkcen(
        RNGAdapter rnga,
        const int [:] ncen,
        const float [:, :] pos,
        const float [:, :] vel, 
        const float [:] vdisp, 
        const float [:] vcen,
        ):

    cdef int igrp
    cdef int icen

    cdef float [:, :] cpos
    cdef float [:, :] cvel

    cdef int Nd = pos.shape[1]

    ninit = numpy.sum(ncen, dtype='i8')

    cpos = numpy.zeros((ninit, Nd), dtype='f4')
    cvel = numpy.zeros((ninit, Nd), dtype='f4')

    icen = 0

    for igrp in range(0, ncen.shape[0]):

        for j in range(ncen[igrp]):
            for i in range(Nd):
                grnd = rnga.normal()
                cpos[icen, i] = pos[igrp, i]
                cvel[icen, i] = vel[igrp, i] + vcen[igrp]*grnd*vdisp[igrp]

            icen = icen + 1

    return numpy.array(cpos), numpy.array(cvel)

def mksat(rng,
    nsat,
    pos, vel, vdisp, conc, rvir,
    vsat=0.5
    ):
    """ Satellite galaxy HOD model

        vdisp is the DM velocity dispersion of the halo.
        conc is the DM halo concentration parameter
        rvir is the radius of the DM halo
 
        vsat is the fraction of velocity relative to the dispersion.

        rng is a random seed or a RandomState object.
        Do not reuse the rng, as the number of draws from the rng
        may change as this code evolves

        Returns (spos, svel), position and velocity of satellites.
    """

    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    rnga = RNGAdapter(rng, min(len(nsat), 1024 * 8))
    vsat = numpy.broadcast_to(vsat, nsat.shape)
    vdisp = numpy.broadcast_to(vdisp, nsat.shape)
    conc = numpy.broadcast_to(conc, nsat.shape)
    rvir = numpy.broadcast_to(rvir, nsat.shape)

    r = _mksat(rnga,
            nsat=nsat.astype('=i4'),
            pos=pos.astype('=f4'),
            vel=vel.astype('=f4'),
            vdisp=vdisp.astype('=f4'),
            conc=conc.astype('=f4'),
            rvir=rvir.astype('=f4'),
            vsat=vsat.astype('=f4'))

    if rnga._rejections > 10 * len(r[0]):
        import warnings
        warnings.warn("Rejection sampling was inefficient: %g/%g samples are used" % (len(r[0]), rnga._rejections), RuntimeWarning)

    return r

cdef _mksat(
        RNGAdapter rnga,
        const int [:] nsat,
        const float [:, :] pos,
        const float [:, :] vel, 
        const float [:] vdisp, 
        const float [:] conc,
        const float [:] rvir,
        const float [:] vsat):

    cdef int igrp
    cdef int isat

    cdef float [:, :] cpos
    cdef float [:, :] cvel
    cdef float dr[3]
    cdef int Nd = pos.shape[1]

    ninit = numpy.sum(nsat, dtype='i8')

    spos = numpy.zeros((ninit, Nd), dtype='f4')
    svel = numpy.zeros((ninit, Nd), dtype='f4')

    isat = 0

    for igrp in range(0, nsat.shape[0]):

        for j in range(nsat[igrp]):
            ctheta= -1 + 2*rnga.drand()
            phi   = 2*PI*rnga.drand()
            rr    = rnga.get_nfw_r(conc[igrp])

            dr[0] = rr*sqrt(1-ctheta*ctheta)*cos(phi)
            dr[1] = rr*sqrt(1-ctheta*ctheta)*sin(phi)
            dr[2] = rr*ctheta;

            for i in range(Nd):
                grnd = rnga.normal()
                spos[isat, i] = pos[igrp, i] + rvir[igrp] * dr[i]
                svel[isat, i] = vel[igrp, i] + vsat[igrp] * grnd * vdisp[igrp]

            isat = isat + 1

    return numpy.array(spos), numpy.array(svel)

cdef class RNGAdapter:
    """
    Sampling from numpy's random number generater from C, 'efficiently'
    """
    cdef long int last
    cdef long int batchsize
    cdef double [:]  buffer
    cdef object rng
    cdef readonly long int _rejections
    def __init__(self, rng, batchsize):
        self.rng = rng
        self.batchsize = batchsize
        self.buffer = rng.uniform(0, 1, size=batchsize)
        self._rejections = 0

    cdef double drand(self):
        cdef double ret  = self.buffer[self.last]
        self.last = self.last + 1
        if self.last == self.buffer.shape[0]:
            self.buffer = self.rng.uniform(0, 1, size=self.batchsize)
            self.last = 0
        return ret

    cdef float normal(self):
        return sqrt(-2*log(self.drand()))*cos(2*PI*self.drand())

    cdef float get_nfw_r(self, float c):
        cdef float x
        # pdf: f(x) = N x /(1+x)**2 = N nfw(x)
        # peaks at xmax;
        # M = bound[ f[y] / g[y] ]
        #   = N nfw(xmax) / g
        # g(y) is constant here.

        cdef float Mg_N
        if c > 1:
            # xmax = 1
            Mg_N = 1/4.
        else:
            # xmax = c
            Mg_N = nfw(c)

        cdef float r
        while True:
            r = self.drand()
            x = r * c;
            self._rejections = self._rejections + 1
            # nfw(x) is rewritten to avoid divisions
            if self.drand() *(1+x) * (1+x) * Mg_N < x:
                return r

cdef float nfw(x):
    return x/((1+x) *(1+x))
