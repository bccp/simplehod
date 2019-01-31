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
    if 'Aemit' in cat.columns:
        redshift = 1 / cat['Aemit'] - 1
    else:
        redshift = 1 / cat.attrs['Time'] - 1
    cat['conc'] = transform.HaloConcentration(cat['Mass'], CP, redshift).compute()
    cat['rvir'] = transform.HaloRadius(cat['Mass'], CP, redshift).compute() / cat.attrs['Time']
    cat['vdisp'] = transform.HaloVelocityDispersion(cat['Mass'], CP, redshift).compute()

    if 'Aemit' not in cat.columns:
        cat['Aemit'] = cat.attrs['Time'][0]

    cat.attrs['BoxSize'] = numpy.ones(3) * cat.attrs['BoxSize'][0] 
    return cat

def _LRGHOD_5p(cat, mcut, sigma, m0, m1, alpha):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_hard_power(cat['Mass'], m0, m1, alpha)
    cat1 = cat.copy()
    ncen, nsat = simplehod.mknint(3333, ncen, nsat)

    nsat = nsat * ncen
    cpos, cvel = simplehod.mkcen(3444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    spos, svel = simplehod.mksat(3454, nsat, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute(), cat['conc'].compute(), cat['rvir'].compute())
    aemit = cat1['Aemit'].compute()
    return numpy.append(cpos, spos, axis=0), numpy.append(cvel, svel, axis=0), \
            numpy.append(numpy.repeat(aemit, ncen), numpy.repeat(aemit, nsat), axis=0)

def _ELGHOD_5p(cat, mcut, sigma, m0, m1, alpha):
    ncen = simplehod.mkn_soft_logstep(cat['Mass'], mcut, sigma)
    nsat = simplehod.mkn_soft_power(cat['Mass'], m0, m1, alpha)
    cat1 = cat.copy()
    ncen, nsat = simplehod.mknint(13333, ncen, nsat)

    nsat = nsat * ncen
    cpos, cvel = simplehod.mkcen(13444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())
    spos, svel = simplehod.mksat(13454, nsat, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute(), cat['conc'].compute(), cat['rvir'].compute())

    aemit = cat1['Aemit'].compute()
    return numpy.append(cpos, spos, axis=0), numpy.append(cvel, svel, axis=0), \
            numpy.append(numpy.repeat(aemit, ncen), numpy.repeat(aemit, nsat), axis=0)



def _QSOHOD_2p(cat, mcen, sigma):
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
 
    cat1 = cat.copy()
    ncen, junk = simplehod.mknint(23333, ncen, None)

    cpos, cvel = simplehod.mkcen(23444, ncen, cat1['Position'].compute(), cat1['Velocity'].compute(), cat1['vdisp'].compute())

    aemit = cat1['Aemit'].compute()
    return cpos, cvel, numpy.repeat(aemit, ncen)

def make_observation(mode, cat, HOD, logrp, params, pimax=None):
    rmax = numpy.nanmax(10**logrp)
    lrg = HOD(cat, *params)
    # update boxsize
    lrg.attrs['BoxSize'] = cat.attrs['BoxSize']
    
    if cat.comm.rank == 0:
        cat.logger.info("HOD produced %d galaxies" % lrg.csize)

    fraction = int(lrg.csize / (0.125 * cat.csize) + 0.5)
    if fraction > 1:
        # subsample to a reasonable load
        lrg = lrg[::fraction]

    lrg['VelocityOffset'] = lrg['Velocity'] * cat.attrs['RSDFactor']
    lrg['Position'] = lrg['Position'] + lrg['VelocityOffset'] * [0, 0, 1]

    if cat.comm.rank == 0:
        cat.logger.info("Sampled to %d galaxies" % lrg.csize)

    edges = numpy.logspace(numpy.log10(0.5), numpy.log10(rmax), 20)
    class TooFewObjects(BaseException): pass

    try:
        # SimulationBox2PCF seems to crash randomly if some ranks have no data,
        # so capture that.
        if any(cat.comm.allgather(lrg.size == 0)):
            print(lrg.size)
            raise TooFewObjects

        if mode == 'projected':
            wlrg = SimulationBox2PCF(data1=lrg, mode='projected', edges=edges, pimax=pimax, show_progress=False).wp
            # out of bound produces nans, which are excluded from the fit.
            logwp_model = numpy.interp(logrp, numpy.log10(wlrg['rp']), numpy.log10(wlrg['corr']), left=numpy.nan, right=numpy.nan)

        if mode == '1d':
            wlrg = SimulationBox2PCF(data1=lrg, mode='1d', edges=edges, show_progress=False).corr
            # out of bound produces nans, which are excluded from the fit.
            logwp_model = numpy.interp(logrp, numpy.log10(wlrg['r']), numpy.log10(wlrg['corr']), left=numpy.nan, right=numpy.nan)

        # if bins are empty, we also bail. Because we do not want to confuse this
        # with interpolation nans.
        if numpy.isnan(numpy.log10(wlrg['corr'])).any():
            print(wlrg['corr'])
            raise TooFewObjects
        
    except TooFewObjects:
        # some huge loss will be caused by this
        logwp_model = logrp * 0.0 - 1.00

    logwp_model = cat.comm.bcast(logwp_model)
    return logwp_model

def fit_wp(cat, pimax, logrp, logwp, HOD, x0, callback):    
    if cat.comm.rank == 0:
        cat.logger.info("scattering by volume, total number = %d" % cat.csize)

    def loss(params):
        logwp_model = make_observation("projected", cat, HOD, logrp, params, pimax=pimax)
        loss = numpy.nansum((logwp - logwp_model)**2) # ignore out of bound values
        loss = cat.comm.bcast(loss)
        return loss
    
    # bfgs wastes too much on gradients, and eps is too big. unstable 
    #return minimize(loss, x0=x0, method='BFGS', callback=print, options={'eps':1e-3 })
    return minimize(loss, x0=x0, method='Nelder-Mead', callback=callback)

def save_cat(filename, dataset, cat, HOD, x):
    gal = HOD(cat, *x)
    
    if cat.comm.rank == 0:
        cat.logger.info("HOD produced %d galaxies" % gal.csize)
        cat.logger.info("Attrs: %s" % str(gal.attrs))
        cat.logger.info("writing to %s %s" % (filename, dataset))

    gal.attrs['BoxSize'] = cat.attrs['BoxSize']
    if 'RSDFactor' in cat.attrs:
        gal.attrs['RSDFactor'] = cat.attrs['RSDFactor']
    gal.save(filename, dataset=dataset, header=dataset)

def save_model(filename, cat, pimax, logrp, logwp, HOD, x, i):

    logwp_model = make_observation("projected", cat, HOD, logrp, x, pimax=pimax)
    logxi_model = make_observation("1d", cat, HOD, logrp, x)

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
            for row in zip(logrp, logwp, logwp_model, logxi_model):
                ff.write("%g %g %g %g \n" % row)

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 3), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(10**logrp, 10**logwp, 'o', label='Reference model')
        ax.plot(10**logrp, 10**logwp_model, 'x', label='Best fit HOD')
        ax.set_ylabel("wp(rp)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax = fig.add_subplot(122)
        ax.plot(10**logrp, 10**logxi_model, 'x', label='Best fit HOD')
        ax.set_ylabel("xi(r)")
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
    
    pos, vel, aemit = _LRGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)

    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)

    return cat2
LRGHOD_5p.x0 = 13.05649281, 0.43743876,13.96182037,13.61134496, 1.00853217 

def LRGHOD_3p(cat, logmcut, logm0, logm1):
    mcut=10**logmcut
    sigma=0.45
    alpha=1.0
    m0=10**logm0
    m1=10**logm1
    
    pos, vel, aemit = _LRGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)

    return cat2
LRGHOD_3p.x0 = 13.05649281, 13.96182037,13.61134496 

def ELGHOD_5p(cat, logmcut, sigma, logm0, logm1, alpha):
    mcut=10**logmcut
    sigma=sigma
    alpha=alpha
    m0=10**logm0
    m1=10**logm1
    
    pos, vel, aemit = _ELGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)

    return cat2
ELGHOD_5p.x0 = 12.0952,0.451427,14.9307,13.8337,0.990796 

def ELGHOD_3p(cat, logmcut, logm0, logm1 ):
    mcut=10**logmcut
    sigma=0.45
    alpha=1.0
    m0=10**logm0
    m1=10**logm1
    
    pos, vel, aemit = _ELGHOD_5p(cat, mcut=mcut, sigma=sigma, alpha=alpha, m0=m0, m1=m1)
    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)

    return cat2
ELGHOD_3p.x0 = 12.0952,14.9307,13.8337

def QSOHOD_1p(cat, logmcen):
    mcen=10**logmcen
    sigma=0.5 * 2.303
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
 
    pos, vel, aemit = _QSOHOD_2p(cat, mcen, sigma)
    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)
    return cat2
QSOHOD_1p.x0 = 12.0,

def QSOHOD_2p(cat, logmcen, sigma):
    mcen=10**logmcen
    sigma=sigma# * 2.303
 
    pos, vel, aemit = _QSOHOD_2p(cat, mcen, sigma)
    cat2 = ArrayCatalog({
        'Position' : pos,
        'Velocity' : vel,
        'Aemit' : aemit,
    }, comm = cat.comm)

    return cat2
QSOHOD_2p.x0 = 12.0, 0.5

############################
# main program
#
import argparse

ap = argparse.ArgumentParser()
# list all models defined about here as strings.
ap.add_argument("model", choices=['LRGHOD_5p', 'LRGHOD_3p', 'ELGHOD_5p', 'ELGHOD_3p', 'QSOHOD_1p', 'QSOHOD_2p'], help='type of model')

sp = ap.add_subparsers(dest='command')

ap1 = sp.add_parser("fit")
ap1.add_argument("--evaluate", help="evaluate the current fit parameter", action="store_true", default=False)
ap1.add_argument("--save", help="save the current fit catalog, filepath is derived from output, and dataset is the value of the argument", default=None)
ap1.add_argument("--pimax", help="max-distance of wp. Some datasets, eboss QSO wp uses very low wp like 60", type=float, default=120.)
ap1.add_argument("output", help='filename to store the best fit parameters')
ap1.add_argument("logwpr", help='filename to the path of table of log rp , log wp(rp) /rp in Mpc/h units')
ap1.add_argument("fastpm", help='path to the fastpm halo catalog')

ap1 = sp.add_parser("apply")
ap1.add_argument("output", help='filename to store the result, BigFileCatalog is written')
ap1.add_argument("--dataset", help='dataest to store the result, default is the model name', default=None, )
ap1.add_argument("bestfits", help='filename pattern to store the best fit parameters. Quote this argument!')
ap1.add_argument("fastpm", help='path to the fastpm halo catalog')


def fit(ns):

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

    cat = cat.to_subvolumes()
    I = [0]
    def callback(x):
        # save every iteration, so if we die, can recover
        save_model(ns.output, cat, ns.pimax, logrp, logwp, HOD, x, I[0])
        if ns.save is not None:
            path = ns.output.rsplit('.', 1)[0]
            save_cat(path, ns.save, cat, HOD, x)

        I[0] = I[0] + 1

    if ns.evaluate:
        # just evaluate, don't fit.
        # e.g. if we updated the parameters externally via smoothing
        # in redshift
        if cat.comm.rank == 0:
            cat.logger.info("evaluating model")
        callback(x0)
    else:
        r = fit_wp(cat, ns.pimax, logrp, logwp, HOD, x0, callback)

    if cat.comm.rank == 0:
        cat.logger.info("finished fitting ")

def read_hod_fits(pattern):
    import glob
    import re
    param_list = []
    a_list = []
    loss_list = []
    for file in sorted(glob.glob(pattern)):
        a = float(re.search('[0-9]\.[0-9][0-9][0-9][0-9]', file).group())
        params = numpy.loadtxt(open(file, 'rb').readlines()[:3])
        params = numpy.atleast_1d(params)
        loss =float(open(file, 'r').readlines()[0].split('=')[1])
        param_list.append(params)
        loss_list.append(loss)
        a_list.append(a)
    a_list = numpy.array(a_list)
    loss_list = numpy.array(loss_list)
    param_list = numpy.array(param_list)
    title_list = [s.strip() for s in  open(file).readlines()[1][1:].split(',')]
    return a_list, param_list, title_list, loss_list

def fit_hodfit(a_list, param_list, title_list, loss_list):
    plist = []
    z_list = 1 / a_list - 1
    for i in range(len(title_list)):
        w = numpy.ones_like(a_list)
        w = 1 / loss_list # [loss_list < numpy.percentile(loss_list, 10.)] = 0
        
        if len(a_list) >= 2:
            p = numpy.polyfit(a_list-0.6667, param_list[:, i], min(2, len(a_list)), w=w)
        else:
            p = param_list[:1, i]
        p = p.round(2)
        plist.append(p)
    return numpy.array(plist)

def apply(ns):
    # get the model. If wrong check if the function is defined.
    HOD = globals()[ns.model]
    if ns.dataset is None:
        ns.dataset = ns.model

    cat = readcat(ns.fastpm)
    a, param, title, loss = read_hod_fits(ns.bestfits)
    p_list = fit_hodfit(a, param, title, loss)

    aemit = cat['Aemit'].compute()

    # generate HOD parameters for each object
    params = []
    aemit_mean = cat.comm.allreduce(aemit.sum(dtype='f8')) / cat.csize
    aemit_std = abs(cat.comm.allreduce((aemit ** 2).sum(dtype='f8')) / cat.csize - aemit_mean ** 2) ** 0.5

    if cat.comm.rank == 0:
        cat.logger.info("Aemit mean = %g std = %g" % (aemit_mean, aemit_std))

    for i, p in enumerate(p_list):
        p1 = numpy.polyval(p, aemit - 0.6667)
        p1_mean = cat.comm.allreduce(p1.sum(dtype='f8')) / cat.csize
        p1_std = abs(cat.comm.allreduce((p1 ** 2).sum(dtype='f8')) / cat.csize - p1_mean ** 2) ** 0.5

        if cat.comm.rank == 0:
            cat.logger.info("%s mean = %g std = %g" % (title[i], p1_mean, p1_std))
        params.append(p1)

    save_cat(ns.output, ns.dataset, cat, HOD, params)

def main():
    ns = ap.parse_args()
    if ns.command == 'fit':
        fit(ns)
    if ns.command == 'apply':
        apply(ns)

if __name__ == '__main__':
    main()
        
