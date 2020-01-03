"""
A few HOD models used by CrowCanyon simulations.

Mostly described by Martin White.

"""
from . import simplehod 

import numpy
from nbodykit.lab import ArrayCatalog

def _mkseed(comm, *args):
    """Create 'unique' seeds on different MPI ranks."""
    return (hash(tuple(args)) % 314159) * comm.size + comm.rank

def _make_balanced_catalog(cat, seed, ncen, nsat):
    ncen, nsat = simplehod.mknint(seed, ncen, nsat)
    if nsat is None:
        nsat = 0
    mask = ncen > 0
    cat1 = cat.copy()
    cat1['ncen'] = ncen
    cat1['nsat'] = nsat
    cat1 = cat1[ncen != 0]
    # load balancing, but keep sorted by Aemit
    cat1.sort(keys=['Aemit'], usecols=['ncen', 'nsat', 'Aemit', 'Position', 'Velocity', 'vdisp', 'conc', 'rvir'])
    total_ncen = cat1.comm.allreduce(cat1['ncen'].sum().compute())
    total_nsat = cat1.comm.allreduce(cat1['nsat'].sum().compute())
    if cat1.comm.rank == 0:
        cat1.logger.info("total number of centrals: %d", total_ncen)
        cat1.logger.info("total number of satellites: %d", total_nsat)
    return cat1

def LRG(cat, mcut, sigma, m0, m1, alpha, tag):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_hard_power(cat['Mass'], m0, m1, alpha)
    cat = _make_balanced_catalog(cat, _mkseed(cat.comm, tag, "mknint"), ncen, nsat)

    cpos, cvel = simplehod.mkcen(_mkseed(cat.comm, tag, "mkcen"),
                cat['ncen'].compute(),
                cat['Position'].compute(),
                cat['Velocity'].compute(),
                cat['vdisp'].compute())
    spos, svel = simplehod.mksat(_mkseed(cat.comm, tag, "mksat"),
                cat['nsat'].compute(),
                cat['Position'].compute(),
                cat['Velocity'].compute(),
                cat['vdisp'].compute(),
                cat['conc'].compute(),
                cat['rvir'].compute())

    ncen = cat['ncen'].compute()
    nsat = cat['nsat'].compute()
    aemit = cat['Aemit'].compute()

    if numpy.isinf(spos).any():
        raise ValueError("found inf in spos")
    if numpy.isinf(svel).any():
        raise ValueError("found inf in svel")

    return ArrayCatalog({
        'Position' : numpy.append(cpos, spos, axis=0), 
        'Velocity' :  numpy.append(cvel, svel, axis=0), 
        'Aemit' :  numpy.append(numpy.repeat(aemit, ncen), numpy.repeat(aemit, nsat), axis=0),
    }, comm = cat.comm)

def UNWISE(cat, mcut, sigma, kappa, m1, alpha, tag):
    """
    Martin White uses this model for UNWISE.

    It is the same as _LRGHOD_5p.
    Martin uses conc=0, but his conc is scaling concentration by exp(conc) = 1, thus no scaling.

    m0 is reparametrized as kappa.
    """


    return LRG(cat, mcut, sigma, kappa * mcut, m1, alpha, tag)

def ELG(cat, mcut, sigma, m0, m1, alpha, tag):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_soft_power(cat['Mass'], m0, m1, alpha)
    cat = _make_balanced_catalog(cat, _mkseed(cat.comm, tag, "mknint"), ncen, nsat)

    cpos, cvel = simplehod.mkcen(_mkseed(cat.comm, tag, "mkcen"),
                        cat['ncen'].compute(),
                        cat['Position'].compute(),
                        cat['Velocity'].compute(),
                        cat['vdisp'].compute(),
                 )
    spos, svel = simplehod.mksat(_mkseed(cat.comm, tag, "mksat"),
                        cat['nsat'].compute(),
                        cat['Position'].compute(),
                        cat['Velocity'].compute(),
                        cat['vdisp'].compute(),
                        cat['conc'].compute(),
                        cat['rvir'].compute(),
                 )

    ncen = cat['ncen'].compute()
    nsat = cat['nsat'].compute()
    aemit = cat['Aemit'].compute()

    return ArrayCatalog({
        'Position' : numpy.append(cpos, spos, axis=0), 
        'Velocity' :  numpy.append(cvel, svel, axis=0), 
        'Aemit' :  numpy.append(numpy.repeat(aemit, ncen), numpy.repeat(aemit, nsat), axis=0),
    }, comm = cat.comm)


def QSO(cat, mcen, sigma, tag):
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
 
    cat = _make_balanced_catalog(cat, _mkseed(cat.comm, tag, "mknint"), ncen, None)

    cpos, cvel = simplehod.mkcen(_mkseed(cat.comm, tag, "mkcen"),
                    cat['ncen'].compute(),
                    cat['Position'].compute(),
                    cat['Velocity'].compute(),
                    cat['vdisp'].compute(),
                 )
    ncen = cat['ncen'].compute()
    aemit = cat['Aemit'].compute()

    return ArrayCatalog({
        'Position' : cpos,
        'Velocity' :  cvel,
        'Aemit' :  numpy.repeat(aemit, ncen),
    }, comm = cat.comm)


