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

from nbodykit.lab import BigFileCatalog, SimulationBox2PCF, ArrayCatalog, RandomCatalog
from nbodykit import cosmology, transform
from nbodykit import setup_logging
import dask.array as da
import numpy

setup_logging()

from scipy.optimize import minimize
import simplehod
from simplehod import crowcanyon

SEED = 34913 # used by all HOD models

def stat(arr, comm):
    csize = comm.allreduce(len(arr))
    mean = comm.allreduce(arr.sum(dtype='f8')) / csize
    std = abs(comm.allreduce((arr ** 2).sum(dtype='f8')) / csize - mean ** 2) ** 0.5
    return mean, std

def readcat(path, subsample=False):
    # Only used to compute concentration etc from halo mass
    # thus no need to be accurate.
    CP = cosmology.Cosmology(Omega0_cdm=0.3, h=0.677, Omega0_b=.05)

    cat = BigFileCatalog(path, dataset='LL-0.200')
    if subsample :
        cat = cat[:subsample]

    M0 = cat.attrs['M0'] * 1e10
    if cat.comm.rank == 0:
        cat.logger.info("mass of a particle %e" % M0)

    cat['Mass'] = cat['Length'] * M0
    if 'Aemit' in cat.columns:
        redshift = 1 / cat['Aemit'] - 1
    else:
        redshift = 1 / cat.attrs['Time'] - 1
    cat['conc'] = transform.HaloConcentration(cat['Mass'], CP, redshift).compute(scheduler="single-threaded")
    # proper to comoving
    cat['rvir'] = transform.HaloRadius(cat['Mass'], CP, redshift).compute() * (1 + redshift)
    cat['vdisp'] = transform.HaloVelocityDispersion(cat['Mass'], CP, redshift).compute()

    if 'Aemit' not in cat.columns:
        cat['Aemit'] = cat.attrs['Time'][0]

    mean, std = stat(cat['Aemit'].compute(), cat.comm)
    if cat.comm.rank == 0:
        cat.logger.info("Aemit mean = %g std = %g" % (mean, std))

    mean, std = stat(cat['Mass'].compute(), cat.comm)
    if cat.comm.rank == 0:
        cat.logger.info("mass mean, std = %g, %g" % (mean, std))

    mean, std = stat(cat['conc'].compute(), cat.comm)
    if cat.comm.rank == 0:
        cat.logger.info("conc mean, std = %g, %g" % (mean, std))

    mean, std = stat(cat['rvir'].compute(), cat.comm)
    if cat.comm.rank == 0:
        cat.logger.info("rvir mean, std = %g, %g" % (mean, std))

    mean, std = stat(cat['vdisp'].compute(), cat.comm)
    if cat.comm.rank == 0:
        cat.logger.info("vdisp mean, std = %g, %g" % (mean, std))

    cat.attrs['BoxSize'] = numpy.ones(3) * cat.attrs['BoxSize'][0] 
    return cat

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
    gal.attrs['BoxSize'] = cat.attrs['BoxSize']

    if cat.comm.rank == 0:
        cat.logger.info("HOD produced %d galaxies" % gal.csize)
        cat.logger.info("Attrs: %s" % str(gal.attrs))
        cat.logger.info("writing to %s %s" % (filename, dataset))

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
    return crowcanyon.LRG(cat, tag=(SEED, "LRGHOD_5p"),
                    mcut=10**logmcut,
                    sigma=sigma,
                    alpha=alpha,
                    m0=10**logm0,
                    m1=10**logm1,
                    )
LRGHOD_5p.x0 = 13.05649281, 0.43743876,13.96182037,13.61134496, 1.00853217 

def LRGHOD_3p(cat, logmcut, logm0, logm1):
    
    return crowcanyon.LRG(cat, tag=(SEED, "LRGHOD_3p"),
                    mcut=10**logmcut,
                    sigma=0.45,
                    alpha=1.0,
                    m0=10**logm0,
                    m1=10**logm1,
                    )
LRGHOD_3p.x0 = 13.05649281, 13.96182037,13.61134496 

def RED_UNWISEHOD_5p(cat, logmcut, sigma, kappa, logm1, alpha):
    return crowcanyon.UNWISE(cat, tag=(SEED, "RED_UNWISEHOD_5p"),
                    mcut=10**logmcut,
                    sigma=sigma,
                    alpha=alpha,
                    kappa=kappa,
                    m1=10**logm1,
                    )
# Some made up numbers
RED_UNWISEHOD_5p.x0 = 11.6, 1.0, 0.1, 13.0, 0.7

def BLUE_UNWISEHOD_5p(cat, logmcut, sigma, kappa, logm1, alpha):
    return crowcanyon.UNWISE(cat, tag=(SEED, "BLUE_UNWISEHOD_5p"), 
                    mcut=10**logmcut,
                    sigma=sigma,
                    alpha=alpha,
                    kappa=kappa,
                    m1=10**logm1,
                    )
# Some made up numbers
BLUE_UNWISEHOD_5p.x0 = 11.6, 1.0, 0.1, 13.0, 0.7

def GREEN_UNWISEHOD_5p(cat, logmcut, sigma, kappa, logm1, alpha):
    return crowcanyon.UNWISE(cat, tag=(SEED, "GREEN_UNWISEHOD_5p"),
                    mcut=10**logmcut,
                    sigma=sigma,
                    alpha=alpha,
                    kappa=kappa,
                    m1=10**logm1,
                    )
# Some made up numbers
GREEN_UNWISEHOD_5p.x0 = 11.6, 1.0, 0.1, 13.0, 0.7


def ELGHOD_5p(cat, logmcut, sigma, logm0, logm1, alpha):
    return crowcanyon.ELG(cat, tag=(SEED, "ELGHOD_5p"),
                    mcut=10**logmcut,
                    sigma=sigma,
                    alpha=alpha,
                    m0=10**logm0,
                    m1=10**logm1,
                    )

ELGHOD_5p.x0 = 12.0952,0.451427,14.9307,13.8337,0.990796 

def ELGHOD_3p(cat, logmcut, logm0, logm1 ):
    return crowcanyon.ELG(cat, tag=(SEED, "ELGHOD_3p"), 
                    mcut=10**logmcut,
                    sigma=0.45,
                    alpha=1.0,
                    m0=10**logm0,
                    m1=10**logm1,
                    )

ELGHOD_3p.x0 = 12.0952,14.9307,13.8337

def QSOHOD_1p(cat, logmcen):
    mcen=10**logmcen
    sigma=0.5 * 2.303
    fcen = 3. # good reason to expect due to lack of kink
    ncen = fcen * simplehod.mkn_lognorm(cat['Mass'], mcen, sigma)
    ncen = ncen * 0.1 # duty cycle max 0.1
    return crowcanyon.QSO(cat, tag=(SEED, "QSOHOD_1p"), mcen=mcen, sigma=sigma)
QSOHOD_1p.x0 = 12.0,

def QSOHOD_2p(cat, logmcen, sigma):
    mcen=10**logmcen
    sigma=sigma# * 2.303
    return crowcanyon.QSO(cat, tag=(SEED, "QSOHOD_2p"), mcen=mcen, sigma=sigma)
QSOHOD_2p.x0 = 12.0, 0.5

############################
# main program
#
# list all models defined about here as strings.
MODELS = ['LRGHOD_5p',
          'LRGHOD_3p',
          'ELGHOD_5p', 
          'ELGHOD_3p',
          'QSOHOD_1p',
          'QSOHOD_2p',
          'RED_UNWISEHOD_5p',
          'GREEN_UNWISEHOD_5p',
          'BLUE_UNWISEHOD_5p',
]

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=3333)
sp = ap.add_subparsers(dest='command')

ap1 = sp.add_parser("fit")
ap1.add_argument("model", choices=MODELS , help='type of model')
ap1.add_argument("--evaluate", help="evaluate the current fit parameter", action="store_true", default=False)
ap1.add_argument("--save", help="save the current fit catalog, filepath is derived from output, and dataset is the value of the argument", default=None)
ap1.add_argument("--pimax", help="max-distance of wp. Some datasets, eboss QSO wp uses very low wp like 60", type=float, default=120.)
ap1.add_argument("output", help='filename to store the best fit parameters')
ap1.add_argument("logwpr", help='filename to the path of table of log rp , log wp(rp) /rp in Mpc/h units')
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
    poly_list = []
    z_list = 1 / a_list - 1
    fitted_param_list = []
    for i in range(len(title_list)):
        w = numpy.ones_like(a_list)
        w = 1 / loss_list # [loss_list < numpy.percentile(loss_list, 10.)] = 0
        
        if len(a_list) >= 2:
            poly = numpy.polyfit(a_list-0.6667, param_list[:, i], min(2, len(a_list)), w=w)
        else:
            poly = param_list[:1, i]
        poly = poly.round(2)
        fitted_param_list.append(numpy.polyval(poly, a_list-0.6667))
        poly_list.append(poly)
    return numpy.array(poly_list), numpy.array(fitted_param_list)

ap1 = sp.add_parser("apply")
ap1.add_argument("model", choices=MODELS , help='type of model')
ap1.add_argument("--subsample", type=int, default=None, help='type of model')
ap1.add_argument("output", help='filename to store the result, BigFileCatalog is written')
ap1.add_argument("--dataset", help='dataest to store the result, default is the model name', default=None, )
ap1.add_argument("bestfits", help='filename pattern to store the best fit parameters. Quote this argument!')
ap1.add_argument("fastpm", help='path to the fastpm halo catalog')

def apply(ns):
    # get the model. If wrong check if the function is defined.
    HOD = globals()[ns.model]

    if ns.dataset is None:
        ns.dataset = ns.model

    a, param, title, loss = read_hod_fits(ns.bestfits)
    poly_list, fitted_param_list = fit_hodfit(a, param, title, loss)

    cat = BigFileCatalog(ns.fastpm, dataset='LL-0.200')
    if cat.comm.rank == 0:
        cat.logger.info("# %s", " ".join(title))
        for i in range(len(a)):
            cat.logger.info(
                    "%05.2f " + " ".join(["%05.2f=%05.2f"] * len(title)),
                   a[i], *sum([[param[i, j], fitted_param_list[j, i]] for j in range(len(title))], []))

    cat = readcat(ns.fastpm, ns.subsample)

    aemit = cat['Aemit'].compute()

    # generate HOD parameters for each object
    params = []

    for i, poly in enumerate(poly_list):
        p1 = numpy.polyval(poly, aemit - 0.6667)
        mean, std = stat(p1, cat.comm)

        if cat.comm.rank == 0:
            cat.logger.info("%s mean = %g std = %g" % (title[i], mean, std))
        params.append(p1)

    save_cat(ns.output, ns.dataset, cat, HOD, params)

ap1 = sp.add_parser("mock")
ap1.add_argument("simcov", choices=['NGP', 'FULL'], help='Sky coverage of the SIMulation. NGP: covering NGP; FULL: covering full sky. Mock will cover full sky.')
ap1.add_argument("output", help='filename to store the result, BigFileCatalog is written')
ap1.add_argument("odataset", help='dataest to store the result, default is the model name', default=None, )
ap1.add_argument("input", help='dataest to read the result, default is the model name', default=None, )
ap1.add_argument("--idataset", help='dataest to read the result, default is the same as the output dataset', default=None, )
ap1.add_argument("nz", help='filename store N ~ Z , format z_low z_high N')
ap1.add_argument("--ncol", type=int, default=2, help='column id of N')

def mock(ns):
    if ns.idataset is None:
        ns.idataset = ns.odataset
    cat = BigFileCatalog(ns.input, dataset=ns.idataset)

    if ns.simcov == 'NGP':
        fsky = 0.5
    elif ns.simcov == 'FULL':
        fsky = 1.0
    else:
        raise

    cat['ZREAL'] = (1 / cat['Aemit'] - 1)

    def compute_va(vel, pos):
        u = pos / (pos **2).sum(axis=-1)[:, None] ** 0.5
        return numpy.einsum('ij,ij->i', vel, u)

    VZ = da.apply_gufunc(compute_va, '(3),(3)->()', cat['Velocity'], cat['Position'])

    C = 299792458. / 1000
    cat['Z'] = (1 + cat['ZREAL']) * (1 + VZ / C) - 1

    zmin, zmax = da.compute(cat['Z'].min(), cat['Z'].max())
 
    zmax = max(cat.comm.allgather(zmax))
    zmin = min(cat.comm.allgather(zmin))

    dNdZ = read_Nz(ns.nz, ns.ncol, zmin, zmax)

    zedges = numpy.linspace(zmin, zmax, 128)
    zcenters = 0.5 * (zedges[:-1] + zedges[1:])

    dNdZ1 = fit_dNdZ(cat, zedges, fsky)

    Z = cat['Z'].compute()
    ntarget = dNdZ(Z)  / dNdZ1(Z) 
    
    ntarget[numpy.isnan(ntarget)] = 0
    #ntarget = ntarget.clip(0, 10)

    rng = numpy.random.RandomState((SEED * 20 + 11) * cat.comm.size + cat.comm.rank)

    if all(cat.comm.allgather((ntarget < 1.0).all())):
        ntarget = rng.binomial(1, ntarget)
    else:
        ntarget = rng.poisson(ntarget)
        if cat.comm.rank == 0:
            cat.logger.info("Up-sampling with poisson because number density is too low")

    pos = cat['Position'].compute().repeat(ntarget, axis=0)
    redshift = cat['Z'].compute().repeat(ntarget, axis=0)
    aemit = cat['Aemit'].compute().repeat(ntarget, axis=0)
    ra, dec = transform.CartesianToEquatorial(pos, frame='galactic')

    if ns.simcov == 'NGP':
        if cat.comm.rank == 0:
            cat.logger.info("Patching the half sky simulation into full sky by flipping z axis")

        ra2, dec2 = transform.CartesianToEquatorial(pos * [1, 1, -1], frame='galactic')

        cat1 = ArrayCatalog({
                'RA' : numpy.concatenate([ra, ra2], axis=0),
                'DEC' : numpy.concatenate([dec, dec2], axis=0),
                'Aemit' : numpy.concatenate([aemit, aemit], axis=0),
                'Z' : numpy.concatenate([redshift, redshift], axis=0),
                }, comm=cat.comm)
    elif ns.simcov == 'FULL':
        cat1 = ArrayCatalog({
                'RA' : ra,
                'DEC' : dec,
                'Aemit' : aemit,
                'Z' : redshift,
                }, comm=cat.comm)

    cat1.save(ns.output, dataset=ns.odataset)

def read_Nz(filename, col, zmin, zmax):
    # use col as the N column
    zlow, zhigh, N = numpy.loadtxt(filename, unpack=True, usecols=(0, 1, col))
    mask = (zlow >= zmin) & (zlow <= zmax)
    mask &= (zhigh >= zmin) & (zhigh <= zmax)

    assert (zlow[1:] == zhigh[:-1]).all()
    zlow = zlow[mask]
    zhigh = zhigh[mask]
    N = N[mask] * 41253

    zedges = numpy.concatenate((zlow[:1], zhigh))
    dNdZ = N / numpy.diff(zedges)
    dNdZ[dNdZ <= 0] = 1e-10
    zcenter = (zlow + zhigh) * 0.5
    zcenter = numpy.concatenate([[zmin], zcenter, [zmax]], axis=0)

    dNdZ = numpy.concatenate([[1e-10], dNdZ, [1e-10]], axis=0)

    spl = behavedspline(zcenter, dNdZ)

    return spl

def behavedspline(x, y):
    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import InterpolatedUnivariateSpline
    y = y.copy()
    y[y<=0] = 1e-10
    spl = UnivariateSpline(x, y, ext='const', k=1, s=len(y) * 10)
    xmin = numpy.min(x) 
    xmax = numpy.max(x) 
    def func(x):
        r = spl(x).copy()
        mask = x < xmin
        mask |= x > xmax
        r[r<0] = 0
        r[mask] = numpy.nan
        return r
    return func

def fit_dNdZ(cat, zedges, fsky):

    Z = cat['Z'].compute()

    h, bins = numpy.histogram(Z, zedges)
    h = cat.comm.allreduce(h)
    zcenter = 0.5 * (zedges[1:] + zedges[:-1])
    h = h / numpy.diff(zedges) / fsky

    dNdZ1 = behavedspline(zcenter, h)

    return dNdZ1


ap1 = sp.add_parser("randoms")
ap1.add_argument("output", help='filename to store the result, BigFileCatalog is written')
ap1.add_argument("odataset", help='dataest to store the result, default is the model name', default=None, )
ap1.add_argument("nz", help='filename store N ~ Z , format z_low z_high N')
ap1.add_argument("--zmax", type=float, default=2.2, help='max redshift to go')
ap1.add_argument("--ncol", type=int, default=2, help='column id of N')
ap1.add_argument("--boost", type=float, default=5, help='boost number density by this factor ')

def geticdf(spl, points):
    vals = spl(points)
    # use middle point
    vals = (vals[1:] + vals[:-1]) * 0.5
    vals[vals<0] = 0
    
    cdf = numpy.cumsum(vals * numpy.diff(points))
    N = cdf[-1]
    cdf /= cdf[-1]
    
    u, ind = numpy.unique(cdf, return_index=True)
    cdf = cdf[ind]
    points = points[ind]
    def interp(x):
        return numpy.interp(x, cdf, points)

    return interp, N

def randoms(ns):
    zmin = 0
    zmax = ns.zmax

    dNdZ = read_Nz(ns.nz, ns.ncol, zmin, zmax)

    zedges = numpy.linspace(zmin, zmax, 12)
    zcenters = 0.5 * (zedges[1:] + zedges[:-1])

    dNdZ_ran = lambda z : dNdZ(z) * ns.boost
    
    icdf, Ntotal = geticdf(dNdZ_ran, numpy.linspace(zmin, zmax, 1024))

    # drop about 20% of samples to give it enough freedom.
    cat = RandomCatalog(
            csize=int(Ntotal),
            seed=SEED * 20 + 19)

    if cat.comm.rank == 0:
        cat.logger.info("Total = %d" % Ntotal)

    # generate points uniformly on a sphere.
    # FIXME: add uniform sphere to nbodykit's RandomCatalog
    z = cat.rng.uniform(-1., 1.)
    r = abs(1 - z ** 2) ** 0.5
    phi = cat.rng.uniform(0, 2 * numpy.pi) 

    x = da.cos(phi) * r
    y = da.sin(phi) * r
    z = z
    
    ra, dec = transform.CartesianToEquatorial(da.stack((x,  y, z), axis=1), frame='galactic')
    
    z = icdf(cat.rng.uniform(0, 1) )

    cat['RA'] = ra.compute()
    cat['DEC'] = dec.compute()
    cat['Z'] = z
    cat['Aemit'] = (1 / (z + 1))

    ntarget = numpy.ones_like(z, dtype='i4')

    randoms = ArrayCatalog({
            'RA' : cat['RA'].compute().repeat(ntarget, axis=0),
            'DEC' : cat['DEC'].compute().repeat(ntarget, axis=0),
            'Z' : cat['Z'].compute().repeat(ntarget, axis=0),
            'Aemit' : cat['Aemit'].compute().repeat(ntarget, axis=0),
            }, comm=cat.comm)
    randoms = randoms.sort(('Aemit',), usecols=['RA', 'DEC', 'Z', 'Aemit'])

    randoms.save(ns.output, dataset=ns.odataset)


def main():
    ns = ap.parse_args()
    global SEED
    SEED = ns.seed

    if ns.command == 'fit':
        fit(ns)
    if ns.command == 'apply':
        apply(ns)
    if ns.command == 'mock':
        mock(ns)
    if ns.command == 'randoms':
        randoms(ns)

if __name__ == '__main__':
    main()
        
