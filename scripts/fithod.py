# fithod.py
#
# Fitting an HOD.
#
# Yu Feng (yfeng1@berkeley.edu)
#
# TODO:
#  - format of wp input is bad -- shall take plain txt files of (rp, wp, wperr)
#    current is log rp, log wp / rp
#
#  - make use of wp error
#  - shall write loss / ddof.
#

from nbodykit.lab import BigFileCatalog, SimulationBox2PCF, ArrayCatalog
from nbodykit import cosmology, transform
from nbodykit import setup_logging
import numpy

setup_logging()

from scipy.optimize import minimize
import simplehod

def readcat(path, subsample=False):
    # Only used to compute concentration etc from halo mass
    # thus no need to be accurate.
    CP = cosmology.Cosmology(Omega0_cdm=0.3, h=0.677, Omega0_b=.05)

    cat = BigFileCatalog(path, dataset='LL-0.200')
    if subsample :
        cat = cat[::subsample]

    M0 = cat.attrs['M0'] * 1e10
    if cat.comm.rank == 0:
        cat.logger.info("mass of a particle %e" % M0)

    cat['Mass'] = cat['Length'] * M0
    redshift = 1 / cat.attrs['Time'] - 1
    cat['conc'] = transform.HaloConcentration(cat['Mass'], CP, redshift).compute()
    cat['rvir'] = transform.HaloRadius(cat['Mass'], CP, redshift).compute() / cat.attrs['Time']
    cat['vdisp'] = transform.HaloVelocityDispersion(cat['Mass'], CP, redshift).compute()
    cat.attrs['BoxSize'] = numpy.ones(3) * cat.attrs['BoxSize'][0] 
    if cat.comm.rank == 0:
        cat.logger.info("scattering by volume, total number = %d" % cat.csize)
    return cat.to_subvolumes()


def _LRGHOD_5p(cat, mcut, sigma, m0, m1, alpha):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_hard_power(cat['Mass'], m0, m1, alpha)
    cat1 = cat.copy()
    ncen, nsat = simplehod.mknint(3333, ncen, nsat)
    nsat = nsat * ncen
    cpos, cvel = simplehod.mkcen(3444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    spos, svel = simplehod.mksat(3454, nsat, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute(), cat['conc'].compute(), cat['rvir'].compute())
    cat2 = ArrayCatalog({
        'Position' : numpy.append(cpos, spos, axis=0)
                    + numpy.append(cvel, svel, axis=0) * cat.attrs['RSDFactor'] * [0, 0, 1.],
    
    }, comm = cat1.comm)
    cat2.attrs.update(cat1.attrs)
    return cat2

def _ELGHOD_5p(cat, mcut, sigma, m0, m1, alpha):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_soft_power(cat['Mass'], m0, m1, alpha)
    cat1 = cat.copy()
    ncen, nsat = simplehod.mknint(3333, ncen, nsat)
    nsat = nsat * ncen
    cpos, cvel = simplehod.mkcen(3444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    spos, svel = simplehod.mksat(3454, nsat, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute(), cat['conc'].compute(), cat['rvir'].compute())
    cat2 = ArrayCatalog({
        'Position' : numpy.append(cpos, spos, axis=0)
                    + numpy.append(cvel, svel, axis=0) * cat.attrs['RSDFactor'] * [0, 0, 1.],
    
    }, comm = cat1.comm)
    cat2.attrs.update(cat1.attrs)
    return cat2
    
def model(cat, HOD, logrp, params):
    rmax = numpy.nanmax(10**logrp)
    lrg = HOD(cat, *params)
    if cat.comm.rank == 0:
        cat.logger.info("HOD produced %d galaxies" % lrg.csize)

    fraction = int(lrg.csize / (0.125 * cat.csize) + 0.5)
    if fraction > 1:
        # subsample to a reasonable load
        lrg = lrg[::fraction]

    if cat.comm.rank == 0:
        cat.logger.info("Sampled to %d galaxies" % lrg.csize)

    edges = numpy.logspace(numpy.log10(0.5), numpy.log10(rmax), 20)
    class TooFewObjects(BaseException): pass

    try:
        # SimulationBox2PCF seems to crash randomly if some ranks have no data,
        # so capture that.
        if any(cat.comm.allgather(lrg.size == 0)):
            raise TooFewObjects

        wlrg = SimulationBox2PCF(data1=lrg, mode='projected', edges=edges, pimax=80, show_progress=False).wp

        # if bins are empty, we also bail. Because we do not want to confuse this
        # with interpolation nans.
        if numpy.isnan(numpy.log10(wlrg['corr'])).any():
            raise TooFewObjects

        # out of bound produces nans, which are excluded from the fit.
        logwp_model = numpy.interp(logrp, numpy.log10(wlrg['rp']), numpy.log10(wlrg['corr']), left=numpy.nan, right=numpy.nan)

    except TooFewObjects:
        # some huge loss will be caused by this
        logwp_model = logrp * 0.0 - 1.00

    logwp_model = cat.comm.bcast(logwp_model)
    return logwp_model

def fit_wp(cat, logrp, logwp, HOD, x0, callback):    
    def loss(params):
        logwp_model = model(cat, HOD, logrp, params)
        loss = numpy.nansum((logwp - logwp_model)**2) # ignore out of bound values
        loss = cat.comm.bcast(loss)
        return loss
    
    # bfgs wastes too much on gradients, and eps is too big. unstable 
    #return minimize(loss, x0=x0, method='BFGS', callback=print, options={'eps':1e-3 })
    return minimize(loss, x0=x0, method='Nelder-Mead', callback=callback)

def save_model(filename, cat, logrp, logwp, HOD, x, i):

    logwp_model = model(cat, HOD, logrp, x)
    loss = numpy.nansum((logwp - logwp_model)**2) # ignore out of bound
    loss = cat.comm.bcast(loss)

    if cat.comm.rank == 0:
        cat.logger.info("Iteration %d: params = %s, loss = %g" % (i, x, loss))

    argnames = ', '.join(HOD.__code__.co_varnames[1:HOD.__code__.co_argcount])
    if cat.comm.rank == 0:
        with open(filename, 'w') as ff:
            ff.write("# best fit loss = %g\n" % loss)
            ff.write("# %s \n" % argnames)
            ff.write("%s \n" % (' '.join(['%g' % p for p in x])))
            ff.write("# log rp, log wp_ref, log wp_bestfit\n")
            for row in zip(logrp, logwp, logwp_model):
                ff.write("%g %g %g\n" % row)

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(4, 3), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(10**logrp, 10**logwp, 'o', label='Reference model')
        ax.plot(10**logrp, 10**logwp_model, 'x', label='Best fit HOD')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        canvas = FigureCanvasAgg(fig)
        fig.savefig(filename.rsplit('.', 1)[0] + '.png')


# define the HOD models
# function signature must be
# function(cat, *params)
# also add an x0 to set the starting point of the fit.
def LRGHOD_5p(cat, logmcut, sigma, logm0, logm1, alpha):
    mcut=10**logmcut
    sigma=sigma
    alpha=alpha
    m0=10**logm0
    m1=10**logm1
    
    lrg = _LRGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    return lrg
LRGHOD_5p.x0 = 13.05649281, 0.43743876,13.96182037,13.61134496, 1.00853217 

def LRGHOD_3p(cat, logmcut, logm0, logm1):
    mcut=10**logmcut
    sigma=0.45
    alpha=1.0
    m0=10**logm0
    m1=10**logm1
    
    lrg = _LRGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    return lrg
LRGHOD_3p.x0 = 13.05649281, 13.96182037,13.61134496 

def ELGHOD_5p(cat, logmcut, sigma, logm0, logm1, alpha):
    mcut=10**logmcut
    sigma=sigma
    alpha=alpha
    m0=10**logm0
    m1=10**logm1
    
    lrg = _ELGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    return lrg
ELGHOD_5p.x0 = 12.0952,0.451427,14.9307,13.8337,0.990796 

def ELGHOD_3p(cat, logmcut, logm0, logm1 ):
    mcut=10**logmcut
    sigma=0.45
    alpha=1.0
    m0=10**logm0
    m1=10**logm1
    
    lrg = _ELGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    return lrg
ELGHOD_3p.x0 = 12.0952,14.9307,13.8337

def QSOHOD_1p(cat, logmcen):
    mcen=10**logmcen
    sigma=0.5 * 2.303
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
 
    cat1 = cat.copy()
    ncen, junk = simplehod.mknint(3333, ncen, None)
    cpos, cvel = simplehod.mkcen(3444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    cat2 = ArrayCatalog({
        'Position' : cpos + cvel * cat.attrs['RSDFactor'] * [0, 0, 1.],
    }, comm = cat1.comm)
    cat2.attrs.update(cat1.attrs)
    return cat2

QSOHOD_1p.x0 = 12.0,

def QSOHOD_2p(cat, logmcen, sigma):
    mcen=10**logmcen
    sigma=sigma# * 2.303
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
 
    cat1 = cat.copy()
    ncen, junk = simplehod.mknint(3333, ncen, None)
    cpos, cvel = simplehod.mkcen(3444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    cat2 = ArrayCatalog({
        'Position' : cpos + cvel * cat.attrs['RSDFactor'] * [0, 0, 1.],
    }, comm = cat1.comm)
    cat2.attrs.update(cat1.attrs)
    return cat2

QSOHOD_2p.x0 = 12.0, 0.5

############################
# main program
#
import argparse

ap = argparse.ArgumentParser()
# list all models defined about here as strings.
ap.add_argument("--model", choices=['LRGHOD_5p', 'LRGHOD_3p', 'ELGHOD_5p', 'ELGHOD_3p', 'QSOHOD_1p', 'QSOHOD_2p'], help='type of model')
ap.add_argument("--evaluate", help="evaluate the current fit parameter", action="store_true", default=False)
ap.add_argument("output", help='filename to store the best fit parameters')
ap.add_argument("logwpr", help='filename to the path of table of log rp , log wp(rp) /rp in Mpc/h units')
ap.add_argument("fastpm", help='path to the fastpm halo catalog')


def main():

    ns = ap.parse_args()
    logrp, logwpr = numpy.loadtxt(ns.logwpr, unpack=True, delimiter=',')
    logwp = logwpr + logrp

    # get the model. If wrong check if the function is defined.
    HOD = globals()[ns.model]

    x0 = HOD.x0

    # try to restart from the latest point:

    cat = readcat(ns.fastpm)

    if cat.comm.rank == 0:
        try:
            x0 = numpy.loadtxt(open(ns.output, 'rb').readlines()[:3], unpack=True)
            x0 = numpy.atleast_1d(x0)
            # just got non-sense?
            if len(x0) != len(HOD.x0):
                cat.logger.info('x0 = %s ' % str(x0))
                x0 = HOD.x0

            cat.logger.info('restarting from parameters x0 %s ' % str(x0))
        except IOError:
            pass
        
    x0 = cat.comm.bcast(x0)

 
    if cat.comm.rank == 0:
        cat.logger.info("start fitting")

    I = [0]
    def callback(x):
        # save every iteration, so if we die, can recover
        save_model(ns.output, cat, logrp, logwp, HOD, x, I[0])
        I[0] = I[0] + 1

    if ns.evaluate:
        # just evaluate, don't fit.
        # e.g. if we updated the parameters externally via smoothing
        # in redshift
        cat.logger.info("evaluating model")
        callback(x0)
    else:
        r = fit_wp(cat, logrp, logwp, HOD, x0, callback)

    if cat.comm.rank == 0:
        cat.logger.info("finished fitting ")

if __name__ == '__main__':
    main()
        
