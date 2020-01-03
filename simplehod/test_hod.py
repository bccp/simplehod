
def test_simplehod():
    from simplehod import hod, mkhodp
    import numpy

    mfof = 10 ** numpy.linspace(10, 15, 20)
    afof = numpy.linspace(0.6, 0.7, len(mfof))

    pos = numpy.random.uniform(size=(len(mfof), 3))
    vel = numpy.random.uniform(size=(len(mfof), 3))

    mcut = mkhodp(afof, 0.6667, -0.1, 10**13.35)
    m1   = mkhodp(afof, 0.6667, -0.1, 10 ** 12.80)

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


def test_verbose():
    from simplehod import mkn_lognorm, mkn_soft_power, mkn_hard_power, mkn_soft_logstep
    from simplehod import mkcen, mksat, mknint
    import numpy

    mfof = 10 ** numpy.linspace(10, 15, 20)

    nlrg_cen = mkn_soft_logstep(mfof, mcut=10**13.14, sigma=0.486 * 0.614)
    nlrg_sat = mkn_soft_power(mfof, m0=10**13.01, m1=10**14.05, alpha=0.97)

    nlrg_sat2 = mkn_hard_power(mfof, m0=10**13.35, m1=10**12.80, alpha=0.8)

    nqso = mkn_lognorm(mfof, mcen=12.4, sigma=0.5 / 2.3)

    for i in zip(nlrg_cen, nlrg_sat, nlrg_sat2, nqso):
        print(i)

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
    test_simplehod()
    #test_nfw()
    test_verbose()
