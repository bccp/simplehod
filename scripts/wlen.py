import nbodykit
from nbodykit.lab import BigFileCatalog
from nbodykit.transform import ConcatenateSources, CartesianToEquatorial
from nbodykit.cosmology import Planck15
import numpy
import bigfile

from mpi4py import MPI
nbodykit.setup_logging()
nbodykit.set_options(dask_chunk_size=1024 * 1024)
nbodykit.set_options(global_cache_size=0)

from nbodykit.utils import DistributedArray, GatherArray

#nbodykit.set_options(global_cache_size=128)

import dask.array as da

# formula

# int dss  Ps(zs) [ int dxl omega_m / sigma(zs, zl) delta_m(t, zl)]

# sukhdeep swapped the integral order

# int dxl delta_m(t, xl) [ int dzs Pl(zs) 1.5 omega_m / sigma(zs, zl)]

# sigma has dimension of L^{-1}

# another convention is to integrate by zl, then the kernel becomes dimensionless, by adding
# a factor of c / H(zl). It was plotted in Sukhdeep 1606.08841 Figure 1 - CMB with zs=1100.

# kappa = int dxl delta_m(tt, xl) X(xl) = int dxl A(xl) n(t, xl) X(xl) / A(xl) / nbar(xl) - int dxl X(xl)

# = 1 / nbar sum X(xl_i) / A(xl_i) - Const.

# A is the area , so size of healpix * xl_i ** 2
# const can be removed by demanding kappa has a mean of zero in the observed region.

def inv_sigma(ds, dl, zl):
    ddls = 1 - numpy.multiply.outer(1 / ds, dl)
    ddls = ddls.clip(0)
    w = (100. / 3e5) ** 2 * (1 + zl)* dl
    inv_sigma_c = (ddls * w)
    return inv_sigma_c
    
    
def wlen(Om, dl, zl, ds, Nzs=1):
    """
        Parameters
        ----------
        dl, zl: distance and redshift of lensing objects
        
        ds: distance source plane bins. if a single scalar, do a delta function bin.
        
        Nzs : number of objects in each ds bin. len(ds) - 1 items
        
    """
    ds = numpy.atleast_1d(ds) # promote to 1d, sum will get rid of it
    integrand = 1.5 * Om * Nzs * inv_sigma(ds, dl, zl)
    Ntot = numpy.sum(Nzs)
    w_lensing = numpy.sum(integrand, axis=0) / Ntot
    
    return w_lensing

from mpl_aea import healpix

def weighted_map(ipix, npix, weights, localsize, comm):
    """ Make a map from particles, for quantities like
    
       W(t) = \int dx delta(t, x) w
       
       Parameters
       ----------
       ipix: array_like
     
       weights : array_like
    
       Returns
       -------
       Wmap, Nmap; distributed maps
       
       Wmap is the weighted map. Nmap is the number of objects
    """

    ipix, labels = numpy.unique(ipix, return_inverse=True)
    N = numpy.bincount(labels)
    weights = numpy.bincount(labels, weights)
    #print("shrink to %d from %d" % (len(ipix), len(labels)))

    del labels
 
    pairs = numpy.empty(len(ipix) + 1, dtype=[('ipix', 'i4'), ('N', 'i4'), ('weights', 'f8') ])
    pairs['ipix'][:-1] = ipix
    pairs['weights'][:-1] = weights
    pairs['N'][:-1] = N

    pairs['ipix'][-1] = npix - 1 # trick to make sure the final length is correct.
    pairs['weights'][-1] = 0
    pairs['N'][-1] = 0

    disa = DistributedArray(pairs, comm=comm)
    disa.sort('ipix')

    w = disa['ipix'].bincount(weights=disa['weights'].local, local=False, shared_edges=False)
    N = disa['ipix'].bincount(weights=disa['N'].local, local=False, shared_edges=False)

    if npix - w.cshape[0] != 0:
        if comm.rank == 0:
            print('padding -- this shouldnt have occured ', npix, w.cshape)
        # pad with zeros, since the last few bins can be empty.
        ipadding = DistributedArray.cempty((npix - w.cshape[0],), dtype='i4', comm=comm)
        fpadding = DistributedArray.cempty((npix - w.cshape[0],), dtype='f8', comm=comm)

        fpadding.local[:] = 0
        ipadding.local[:] = 0

        w = DistributedArray.concat(w, fpadding)
        N = DistributedArray.concat(N, ipadding)

    w = DistributedArray.concat(w, localsize=localsize)
    N = DistributedArray.concat(N, localsize=localsize)

    return w.local, N.local
   


def read_range(cat, amin, amax):
    edges = cat.attrs['aemitIndex.edges']
    offsets = cat.attrs['aemitIndex.offset']
    start, end = edges.searchsorted([amin, amax])
    if cat.comm.rank == 0:
        cat.logger.info("Range of index is %d to %d" %(( start + 1, end + 1)))
    start = offsets[start + 1]
    end = offsets[end + 1]
    cat =  cat.query_range(start, end)
    cat1 = cat.copy()
    cat1['Position'] = cat1['Position'] * [1, 1, -1.]
    cat3 = ConcatenateSources(cat, cat1)
    if cat1.csize > 0:
        cat3['RA'], cat3['DEC'] = CartesianToEquatorial(cat3['Position'], frame='galactic')
    else:
        cat3['RA'] = 0
        cat3['DEC'] = 0
    return cat3

def make_kappa_maps(cat, nside, zs_list, ds_list, localsize, nbar):
    """ Make kappa maps at a list of ds

        Return kappa, Nm in shape of (n_ds, npix), kappabar in shape of (n_ds,)

    """

    dl = (abs(cat['Position'] **2).sum(axis=-1)) ** 0.5
    chunks = dl.chunks
    ra = cat['RA']
    dec = cat['DEC']
    zl = (1 / cat['Aemit'] - 1)
    
    ipix = da.apply_gufunc(lambda ra, dec, nside:
                           healpix.ang2pix(nside, numpy.radians(90 - dec), numpy.radians(ra)),
                        '(),()->()', ra, dec, nside=nside)

    npix = healpix.nside2npix(nside)

    ipix = ipix.compute()
    dl = dl.persist()
 
    cat.comm.barrier()

    if cat.comm.rank == 0:
        cat.logger.info("ipix and dl are persisted")

    area = (4 * numpy.pi / npix) * dl**2

    Om = cat.attrs['OmegaM'][0]
    
    kappa_list = []
    kappabar_list = []
    Nm_list = []
    for zs, ds in zip(zs_list, ds_list):
        LensKernel = da.apply_gufunc(lambda dl, zl, Om, ds: wlen(Om, dl, zl, ds), 
                                     "(), ()-> ()",
                                     dl, zl, Om=Om, ds=ds)

        weights = (LensKernel / (area * nbar))
        weights = weights.compute()

        cat.comm.barrier()

        if cat.comm.rank == 0:
            cat.logger.info("source plane %g weights are persisted" % zs)
        Wmap, Nmap = weighted_map(ipix, npix, weights, localsize, cat.comm)

        cat.comm.barrier()
        if cat.comm.rank == 0:
            cat.logger.info("source plane %g maps generated" % zs)

        # compute kappa bar
        # this is a simple integral, but we do not know dl, dz relation
        # so do it with values from a subsample of particles
        every = (cat.csize // 100000)
        
        kappa1 = Wmap
        if every == 0: every = 1
        ssdl = GatherArray(dl[::every].compute(), cat.comm)
        ssLensKernel = GatherArray(LensKernel[::every].compute(), cat.comm)

        if cat.comm.rank == 0:
            arg = ssdl.argsort()
            ssdl = ssdl[arg]
            ssLensKernel = ssLensKernel[arg]
            
            kappa1bar = numpy.trapz(ssLensKernel, ssdl)
        else:
            kappa1bar = None
        kappa1bar = cat.comm.bcast(kappa1bar)

        cat.comm.barrier()
        if cat.comm.rank == 0:
            cat.logger.info("source plane %g bar computed " % zs)
        kappa_list.append(kappa1)
        kappabar_list.append(kappa1bar)
        Nm_list.append(Nmap)
    """
    # estimate nbar
    dlmin = dl.min()
    dlmax = dl.max()
        
    volume = (Nmap > 0).sum() / len(Nmap) * 4  / 3 * numpy.pi * (dlmax**3 - dlmin ** 3)
    """
    # returns number rather than delta, since we do not know fsky here.
    #Nmap = Nmap / cat.csize * cat.comm.allreduce((Nmap > 0).sum()) # to overdensity.
    return numpy.array(kappa_list), numpy.array(kappabar_list), numpy.array(Nm_list)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('output')
ap.add_argument('source')
ap.add_argument('zs', nargs='+', type=float)
ap.add_argument('--dataset', default='1')
ap.add_argument('--zlmin', type=float, default=0.00)
ap.add_argument('--zlmax', type=float, default=None)
ap.add_argument('--zstep', type=float, default=0.5)
ap.add_argument('--nside', type=int, default=512)

def main(ns):
    if ns.zlmax is None:
        ns.zlmax = max(ns.zs)

    zs_list = ns.zs

    zlmin = ns.zlmin
    zlmax = ns.zlmax

    # no need to be accurate here
    ds_list = Planck15.comoving_distance(zs_list)

    path = ns.source
    #'/global/cscratch1/sd/yfeng1/m3127/desi/1536-9201-40eae2464/lightcone/usmesh/'

    cat = BigFileCatalog(path, dataset=ns.dataset)

    kappa = 0
    Nm = 0
    kappabar = 0

    npix = healpix.nside2npix(ns.nside)
    localsize = npix * (cat.comm.rank + 1) // cat.comm.size - npix * (cat.comm.rank) // cat.comm.size
    nbar = (cat.attrs['NC'] ** 3  / cat.attrs['BoxSize'] ** 3 * cat.attrs['ParticleFraction'])[0]

    Nsteps = int(numpy.round((zlmax - zlmin) / ns.zstep))
    if Nsteps < 2 : Nsteps = 2
    z = numpy.linspace(zlmax, zlmin, Nsteps, endpoint=True)

    if cat.comm.rank == 0:
        cat.logger.info("Splitting data redshift bins %s" % str(z))

    for z1, z2 in zip(z[:-1], z[1:]):
        import gc
        gc.collect()
        if cat.comm.rank == 0:
            cat.logger.info("nbar = %g, zlmin = %g, zlmax = %g zs = %s" % (nbar, z2, z1, zs_list))

        slice = read_range(cat, 1/(1 + z1), 1 / (1 + z2))

        if slice.csize == 0: continue
        if cat.comm.rank == 0:
            cat.logger.info("read %d particles" % slice.csize)

        kappa1, kappa1bar, Nm1  = make_kappa_maps(slice, ns.nside, zs_list, ds_list, localsize, nbar)

        kappa = kappa + kappa1
        Nm = Nm + Nm1
        kappabar = kappabar + kappa1bar

    cat.comm.barrier()

    if cat.comm.rank == 0:
        # use bigfile because it allows concurrent write to different datasets.
        cat.logger.info("writing to %s", ns.output)


    for i, (zs, ds) in enumerate(zip(zs_list, ds_list)):
        std = numpy.std(cat.comm.allgather(len(kappa[i])))
        mean = numpy.mean(cat.comm.allgather(len(kappa[i])))
        if cat.comm.rank == 0:
            cat.logger.info("started gathering source plane %s, size-var = %g, size-bar = %g" % (zs, std, mean))

        kappa1 = GatherArray(kappa[i], cat.comm)
        Nm1 = GatherArray(Nm[i], cat.comm)

        if cat.comm.rank == 0:
            cat.logger.info("done gathering source plane %s" % zs)

        if cat.comm.rank == 0:
            fname = ns.output + "/WL-%02.2f-N%04d" % (zs, ns.nside)
            cat.logger.info("started writing source plane %s" % zs)

            with bigfile.File(fname, create=True) as ff:

                ds1 = ff.create_from_array("kappa", kappa1, Nfile=1)
                ds2 = ff.create_from_array("Nm", Nm1, Nfile=1)

                for d in ds1, ds2:
                    d.attrs['kappabar'] = kappabar[i]
                    d.attrs['nside'] = ns.nside
                    d.attrs['zlmin'] = zlmin
                    d.attrs['zlmin'] = zlmax
                    d.attrs['zs'] = zs
                    d.attrs['ds'] = ds
                    d.attrs['nbar'] = nbar

        cat.comm.barrier()
        if cat.comm.rank == 0:
            # use bigfile because it allows concurrent write to different datasets.
            cat.logger.info("source plane at %g written. " % zs)
    #        numpy.savez(ns.output, kappa1=kappa1, kappa1bar=kappa1bar, deltam=deltam, zlmin=zlmin, zlmax=zlmax, zs=zs, ds=ds)

if __name__ == '__main__':
    ns = ap.parse_args()
    main(ns)
