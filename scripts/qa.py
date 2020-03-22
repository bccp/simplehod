"""
QA of HOD catalogs.

This script computes a bunch of statistics on mock galaxy catalogs and plot them.

- power spectrum multipole (l=0, 2, 4)

- correlation function multipole  (with downsampling for speed, l=0, 2, 4)

- redshift distribution

- object surface density

It takes a bigfile catalog, and two arguments for the data and random objects to use.

A fiducial Planck15 cosmology is used to convert RA, DEC, Z to position .

"""
import nbodykit

nbodykit.setup_logging()
import numpy
from mpl_aea import healpix

from nbodykit.lab import FKPCatalog, ConvolvedFFTPower, RedshiftHistogram, BigFileCatalog, SurveyData2PCF

from nbodykit.cosmology import Planck15
from nbodykit.transform import SkyToCartesian
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('output')
ap.add_argument('cat')
ap.add_argument('data')
ap.add_argument('rand')

def main(ns):
    data = BigFileCatalog(ns.cat, dataset=ns.data)
    rand = BigFileCatalog(ns.cat, dataset=ns.rand)

    dndz_data = RedshiftHistogram(data, fsky=1.0, cosmo=Planck15, redshift='Z')

    ssdata = data.persist(['RA', 'DEC', 'Z'])
    ssrand = rand.persist(['RA', 'DEC', 'Z'])

    dndz_data = RedshiftHistogram(ssdata, fsky=1, cosmo=Planck15, redshift='Z')
    dndz_rand = RedshiftHistogram(ssrand, fsky=1, cosmo=Planck15, redshift='Z')

    hmap = healpix.histogrammap(ssdata['RA'].compute(), ssdata['DEC'].compute(), nside=128, perarea=True)
    data_hmap = data.comm.allreduce(hmap.astype('f8'))

    hmap = healpix.histogrammap(ssrand['RA'].compute(), ssrand['DEC'].compute(), nside=128, perarea=True)
    rand_hmap = data.comm.allreduce(hmap.astype('f8'))


    ssdata['NZ'] = dndz_data.interpolate(ssdata['Z'])
    ssrand['NZ'] = dndz_data.interpolate(ssrand['Z'])

    ssdata['Position'] = SkyToCartesian(ssdata['RA'], ssdata['DEC'], ssdata['Z'], Planck15)
    ssrand['Position'] = SkyToCartesian(ssrand['RA'], ssrand['DEC'], ssrand['Z'], Planck15)

    cat = FKPCatalog(data=ssdata, randoms=ssrand,)

    fkp = ConvolvedFFTPower(cat, poles=[0, 2, 4], Nmesh=512*4)
    basename, ext = ns.output.rsplit('.', 1)
    fkp.save(basename + '-fkp' + '.' + ext)

    edges = numpy.arange(4, 140, 4)
    everydata = max(int(ssdata.size  / 40000 + 0.5), 1)
    everyrand = max(int(ssrand.size  / 40000 + 0.5), 1)

    xir = SurveyData2PCF(mode='2d', data1=ssdata[::everydata], randoms1=ssrand[::everyrand],
            cosmo=Planck15, redshift='Z', edges=edges, Nmu=10,
            show_progress=True)
    xir.save(basename + '-xir' + '.' + ext)

    if data.comm.rank == 0:
        data_hmap.tofile(basename + 'data-hmap.f8')
        rand_hmap.tofile(basename + 'rand-hmap.f8')

        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(6, 12), dpi=200)
        gs = GridSpec(5, 2)

        for i, l in enumerate([0, 2, 4]):
            ax = fig.add_subplot(gs[i, 0])

            ax.plot(fkp.poles['k'], fkp.poles[f'power_{l}'] - fkp.attrs['shotnoise'], label=f'l={l}')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('k [h/Mpc]')
            ax.set_ylabel('P(k)')
            ax.grid()
            ax.legend()

        poles = xir.corr.to_poles([0, 2, 4])
        for i, l in enumerate([0, 2, 4]):
            ax = fig.add_subplot(gs[i, 1])
            ax.plot(poles['r'], poles['r'] ** 2 * poles[f'corr_{l}'], label=f'l = {l}')
            ax.set_xlabel('r [Mpc/h]')
            ax.set_ylabel('r^2 xi(r)')
            ax.grid()
            ax.legend()
            ax.set_title('Downsample D%d R%d' % (everydata, everyrand))

        ax = fig.add_subplot(gs[3, 0])

        ax.plot(dndz_data.bin_centers, dndz_data.nbar, label='n(z) data')
        ax.plot(dndz_rand.bin_centers, dndz_rand.nbar * (data.csize /rand.csize), label='n(z) * D / R random')
        ax.grid()
        ax.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('n(z)')

        ax = fig.add_subplot(gs[3, 1])
        
        ax.hist(data_hmap, histtype='step', label='n(A) data', log=True)
        ax.hist(rand_hmap * data.csize / rand.csize, histtype='step', label='n(A) * D / R random', log=True)
        ax.set_xlabel("N per sqdeg")
        ax.set_ylabel('counts')
        ax.legend()
        ax.grid()

        ax = fig.add_subplot(gs[4, 0],  projection="ast.mollweide")
        ax.mapshow(data_hmap)
        ax.set_title("data")

        ax = fig.add_subplot(gs[4, 1],  projection="ast.mollweide")
        ax.mapshow(rand_hmap)
        ax.set_title("randoms")

        canvas = FigureCanvasAgg(fig)
        fig.tight_layout() 
        fig.savefig(basename + '.png', bbox_inches='tight')
 
if __name__ == '__main__':

    ns = ap.parse_args()
    main(ns)
