
def test_simplehod():
    from simplehod import hod, mkhodp_linear
    import numpy

    mfof = 10 ** numpy.linspace(10, 15, 20)
    afof = numpy.linspace(0.6, 0.7, len(mfof))

    pos = numpy.random.uniform(size=(len(mfof), 3))
    vel = numpy.random.uniform(size=(len(mfof), 3))

    dMda = -0.1

    mcut, m1 = mkhodp_linear(afof, apiv=0.6667, hod_dMda=-0.1, mcut_apiv=10**13.35, m1_apiv=10 ** 12.80)

    (ncen, cpos, cvel), (nsat, spos, svel) = hod(3333,
                        mfof, pos, vel,
                        conc=7,
                        rvir=3,
                        vdisp=1100,
                        mcut=mcut,
                        m1=m1,
                        sigma=0.25,
                        kappa=1.0,
                        alpha=0.8,
                        vcen=0,
                        vsat=0.5)

    hid = numpy.repeat(range(len(mfof)), ncen)
    print(cpos - pos[hid])
    print(cvel - vel[hid])

    hid = numpy.repeat(range(len(mfof)), nsat)
    print(spos - pos[hid])
    print(svel - vel[hid])

    from sys import stdout

    numpy.savetxt(stdout,
        numpy.array([mfof, afof, ncen, nsat]).T,
    header = "mfof, afof, ncen, nsat")

def test_nfw():
    from simplehod import mksat
    import numpy

    nsat = numpy.array([100000], dtype='int')
    vdisp = 1000
    conc = 0.5
    vsat = 0.5
    pos = numpy.zeros(shape=(1, 3))
    vel = numpy.zeros(shape=(1, 3))
    rvir = 1

    spos, svel = mksat(0, nsat, pos, vel, vdisp, conc, rvir, vsat)

    r = (spos **2).sum(axis=-1)**0.5

    def gsum(x):
        return numpy.log(1+x) - x/(1+x)

    def pdf(x, c):
        y = c * x
        nfw = y * (1+y)**-2
        norm = gsum(c)
        return c * nfw / norm

    h, edges = numpy.histogram(r, range=(0, 1), bins=100)
    centers = 0.5*(edges[1:] + edges[:-1])
    pdftest = h
    pdfexp  = pdf(centers, conc) * numpy.diff(edges) * len(r)
    # poisson sample
    assert(abs(numpy.std((pdftest - pdfexp) / pdfexp ** 0.5) - 1.0) < 0.05)
    
if __name__ == '__main__':
    #test_simplehod()
    test_nfw()
