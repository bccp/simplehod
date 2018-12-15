
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

if __name__ == '__main__':
    test_simplehod()
