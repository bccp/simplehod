#cython: language_level=2, boundscheck=False
#cython: embedsignature=True

"""

The verbose interface:

mkhodp : redshift evolution of mass parameters
mknint : sampling from expected number of galaxies
mkn_soft_logstep : expected number from a soft logstep function
mkn_lognorm : expected number from a log-normal
mkn_soft_power : expected number from a power law function with soft cut off
mkn_hard_power : expected number from a soft power law function hard cut off.
"""
import numpy
from libc.math cimport log, sqrt, sin, cos, erf, exp
from libc.math cimport M_PI as PI
from numpy.random import RandomState

def mkhodp(afof, apiv, dlnM_da, *args):
    """ Generate log-linearly evolving HOD mass parameters based
        on values at a pivot, a=apiv, for halos at scale factor afof.

        The parameters are provided as an ordered list on the arguments,
        and evolved parameters are returned in the same order.

        Each parameter must be mass-like.

        Parameters
        ----------
        dlnM_da : float
            To convert to d log M / da, divide by 2.3 (which is ln 10).

        A typical value suggested by Martin for LRG is apiv=0.6667, dlnM_da=-0.1

    """
    dm = (afof - apiv) * dlnM_da # relative fraction.
    rt = [ mparam * (1 + dm) for mparam in args ]
    if len(args) == 1:
        return rt[0]
    return rt

def mknint(rng, ncen, nsat):
    """ generate integer samples of ncen and nsat.

        This step is needed after mkn if you want the exact
        number of objects.

        ncen is draw from a binomial with n = 1 (thus we
        at most have 1 ncen)

        nsat is draw from a poisson.

        if None, do not draw integer for that type.
    """

    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    if ncen is not None:
        ncen = rng.binomial(1, ncen)

    if nsat is not None:
        nsat = rng.poisson(nsat)

    return ncen, nsat

def mkn_lognorm(mfof, mcen, sigma):
    r"""
    Compute the expected number of galaxies from log-normal transformation.

    The lognormal uses ln, not lg for transformation.

    The second equal sign converts to the form used in
    arxiv: 1212.4526 (eq 10)

    .. math ::

        N_\mathrm{cen} = exp ( -0.5 [\frac{ln M - ln Mcen}{\sigma}]^2 )
                       = exp ( -0.5 [\frac{log M - log Mcen}{\sigma / ln 10}]^2 )

    We do not model the duty cycle parameter f_cen in ths function.

    """
    mfof, mcen, sigma = numpy.broadcast_arrays(
            numpy.array(mfof),
            numpy.array(mcen),
            numpy.array(sigma),
            )

    return _mkn_lognorm(
            mfof=mfof.astype('=f4'),
            mcen=mcen.astype('=f4'),
            sigma=sigma.astype('=f4'),
           )

cdef _mkn_lognorm(
        const float [:] mfof,
        const float [:] mcen,
        const float [:] sigma,
):

    cdef int igrp

    cdef float [:] ncen

    ncen = numpy.zeros(mfof.shape[0], dtype='f4')

    cdef float mass, logm, mu

    with nogil:
        for igrp in range(0, mfof.shape[0]):
            mass = mfof[igrp]
            # watch out: literature may use log10 here which
            # means a different sigma definition!
            logm = log(mass / mcen[igrp]) 

            if logm > - 5 * sigma[igrp]:
                mu = exp(-0.5 * (logm / sigma[igrp]) **2)
                ncen[igrp] = mu

    return numpy.array(ncen)

def mkn_soft_logstep(mfof, mcut, sigma):
    """
    Compute the expected number of galaxies from soft_logstep.

    The log transformation uses ln, not lg for transformation.

    The second equal sign converts to the form used in
    arxiv: 1212.4526 (eq 8, eq 9)

    .. math ::

        N_\mathrm{cen} = 0.5 ( 1 +
                    erf(\frac{ln M - ln M_{cut}}{\sqrt{2} \sigma})
                    )

                    = 0.5 ( 1 + 
                    erf(\frac{log M - log M_{cut}}{0.614 \sigma})
                    )

    """
    mfof, mcut, sigma = numpy.broadcast_arrays(
            numpy.array(mfof),
            numpy.array(mcut),
            numpy.array(sigma),
            )

    return _mkn_soft_logstep(
            mfof=mfof.astype('=f4'),
            mcut=mcut.astype('=f4'),
            sigma=sigma.astype('=f4'),
           )

cdef _mkn_soft_logstep(
        const float [:] mfof,
        const float [:] mcut,
        const float [:] sigma,
):

    cdef int igrp

    cdef float [:] ncen

    ncen = numpy.zeros(mfof.shape[0], dtype='f4')

    cdef float mass, logm, mu

    with nogil:
        for igrp in range(0, mfof.shape[0]):

            mass = mfof[igrp]
            # watch out: literature may use log10 here which
            # means a different sigma definition!
            logm = log(mass / mcut[igrp]) 

            ncen[igrp] = 0

            if sigma[igrp] <= 0:
                if mass > mcut[igrp] :
                    ncen[igrp] = 1
            else:
                if logm > -5 * sigma[igrp]:
                    mu = 0.5*(1+erf(logm/sqrt(2.0)/sigma[igrp]))
                    ncen[igrp] = mu

    return numpy.array(ncen)

def mkn_hard_power(mfof, m0, m1, alpha):
    r"""
    Compute number of galaxies from a powerlaw with a sharp cut off

    .. math ::

        N_\mathrm{sat} = (\frac{M - M_{0}){M_1})^{\alpha}

                       = exp(- \alpha \frac{M_{0}}{M})
                           (\frac{M}{M_1})^\alpha

    This is similiar to the powerlaw with a rolloff, as we see
    in the second equal sign.
    """

    mfof, m0, m1, alpha = numpy.broadcast_arrays(
            numpy.array(mfof),
            numpy.array(m0),
            numpy.array(m1),
            numpy.array(alpha),
           )

    return _mkn_hard_power(
            mfof=mfof.astype('=f4'),
            m0=m0.astype('=f4'),
            m1=m1.astype('=f4'),
            alpha=alpha.astype('=f4'),
           )

cdef _mkn_hard_power(
        const float [:] mfof,
        const float [:] m0,
        const float [:] m1,
        const float [:] alpha,
):
    cdef int igrp

    cdef float [:] nsat

    nsat = numpy.zeros(mfof.shape[0], dtype='f4')

    cdef float mass, mu

    with nogil:
        for igrp in range(0, mfof.shape[0]):

            mass = mfof[igrp]

            # sats. similar to the exponential form,
            # but with a sharp cut-off.
            if mass > m0[igrp]:
                mu = ((mass-m0[igrp])/m1[igrp]) ** alpha[igrp]
                nsat[igrp] = mu

    return numpy.array(nsat)

def mkn_soft_power(mfof, m0, m1, alpha):
    r"""
    Compute number of galaxies from a powerlaw with a roll cut off

    .. math ::

        N_\mathrm{sat} = exp(- \frac{M_0}{M})
                           (\frac{M}{M_1})^\alpha

    """

    mfof, m0, m1, alpha = numpy.broadcast_arrays(
            numpy.array(mfof),
            numpy.array(m0),
            numpy.array(m1),
            numpy.array(alpha),
            )

    return _mkn_soft_power(
            mfof=mfof.astype('=f4'),
            m0=m0.astype('=f4'),
            m1=m1.astype('=f4'),
            alpha=alpha.astype('=f4'),
           )

cdef _mkn_soft_power(
        const float [:] mfof,
        const float [:] m0,
        const float [:] m1,
        const float [:] alpha,
):
    cdef int igrp

    cdef float [:] nsat

    nsat = numpy.zeros(mfof.shape[0], dtype='f4')

    cdef float mass, mu

    with nogil:
        for igrp in range(0, mfof.shape[0]):
            mass = mfof[igrp]
            mu = exp(- m0[igrp] / mass) * (mass / m1[igrp]) ** alpha[igrp]
            nsat[igrp] = mu

    return numpy.array(nsat)

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

    If input data is already halo properties per central, pass
    ncen = numpy.ones(len(pos), dtype='i4')

    Returns (cpos, cvel), position and velocity of centrals.

    """
    if not isinstance(rng, RandomState):
        rng = RandomState(rng)
    vcen = numpy.broadcast_to(vcen, ncen.shape)
    vdisp = numpy.broadcast_to(vdisp, ncen.shape)
    
    rnga = RNGAdapter(rng, min(len(ncen), 1024 * 1024))

    return _mkcen(rnga,
            ncen=numpy.array(ncen.astype('=i4')),
            pos=numpy.array(pos.astype('=f4')),
            vel=numpy.array(vel.astype('=f4')),
            vdisp=numpy.array(vdisp.astype('=f4')),
            vcen=numpy.array(vcen.astype('=f4')),
           )

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
    cdef int Ndv = vel.shape[1]

    ninit = numpy.sum(ncen, dtype='i8')

    cpos = numpy.zeros((ninit, Nd), dtype='f4')
    cvel = numpy.zeros((ninit, Ndv), dtype='f4')

    icen = 0

    cdef float grnd

    cdef int j, i

    with nogil:
        for igrp in range(0, ncen.shape[0]):

            for j in range(ncen[igrp]):
                for i in range(Nd):
                    cpos[icen, i] = pos[igrp, i]

                for i in range(Ndv):
                    grnd = rnga.normal()
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

        If input data is already halo properties per satellites, pass
        nsat = numpy.ones(len(pos), dtype='i4')

        Returns (spos, svel), position and velocity of satellites.
    """

    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    rnga = RNGAdapter(rng, min(len(nsat), 1024 * 1024))
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

    cdef float [:, :] spos
    cdef float [:, :] svel
    cdef float dr[3]
    cdef int Nd = pos.shape[1]
    cdef int Ndv = vel.shape[1]

    ninit = numpy.sum(nsat, dtype='i8')

    spos = numpy.zeros((ninit, Nd), dtype='f4')
    svel = numpy.zeros((ninit, Ndv), dtype='f4')

    isat = 0

    cdef float grnd, ctheta, phi, rr

    cdef int j, i
    with nogil:
        for igrp in range(0, nsat.shape[0]):

            for j in range(nsat[igrp]):
                ctheta= -1 + 2*rnga.drand()
                phi   = 2*PI*rnga.drand()
                rr    = rnga.get_nfw_r(conc[igrp])

                dr[0] = rr*sqrt(1-ctheta*ctheta)*cos(phi)
                dr[1] = rr*sqrt(1-ctheta*ctheta)*sin(phi)
                dr[2] = rr*ctheta;

                for i in range(Nd):
                    spos[isat, i] = pos[igrp, i] + rvir[igrp] * dr[i]

                for i in range(Ndv):
                    grnd = rnga.normal()
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

    cdef double drand(self) nogil:
        cdef double ret  = self.buffer[self.last]
        self.last = self.last + 1
        if self.last == self.buffer.shape[0]:
            with gil:
                self.buffer = self.rng.uniform(0, 1, size=self.batchsize)
            self.last = 0
        return ret

    cdef float normal(self) nogil:
        return sqrt(-2*log(self.drand()))*cos(2*PI*self.drand())

    cdef float get_nfw_r(self, float c) nogil:
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
        elif c == 0:
            return 0
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

cdef float nfw(float x) nogil:
    return x/((1+x) *(1+x))

def hod(rng, mfof,
    pos=None, vel=None, conc=None, rvir=None, vdisp=None,
    mcut=10**13.35, sigma=0.25, m1=10**12.80, kappa=1.0,
    alpha=0.8, vcen=0.0, vsat=0.5
    ):
    """
    Apply vanilla HOD to a halo catalog.

    This uses mkn_soft_logstep for centrals, and
    mkn_hard_power for satellites.

    For more flexible models, use the verbose version of API

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

    ncen = mkn_soft_logstep(
            mfof=mfof,
            mcut=mcut, sigma=sigma)

    nsat = mkn_hard_power(
            mfof=mfof,
            m0=mcut * kappa, m1=m1,
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



# Deprecated functions
def mkhodp_linear(afof, apiv, hod_dMda, mcut_apiv, m1_apiv):
    import warnings
    warnings.warn("Use mkhodp instead of this function.", DeprecationWarning, stacklevel=2)
    return mkhodp(afof, apiv, hod_dMda, mcut_apiv, m1_apiv)

def mkn(mfof,
    mcut=10**13.35, sigma=0.25, m1=10**12.8, kappa=1.0,
    alpha=0.8):
    r"""
    This function is deprecataed. Use explict mkn_soft_logstep or mkn_hard_power instead.
    """
    import warnings
    warnings.warn("Use mkn_soft_logstep and mkn_hard_power instead.", DeprecationWarning, stacklevel=2)

    ncen = mkn_soft_logstep(mfof=mfof, mcut=mcut, sigma=sigma)

    nsat = mkn_hard_power(mfof=mfof,
            m0=mcut * kappa,
            m1=m1,
            alpha=alpha)

    return ncen, nsat

