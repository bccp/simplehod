from simplehod import crowcanyon
import numpy

def test_naive():
    from nbodykit.lab import ArrayCatalog

    cat = ArrayCatalog({
       'Mass' : numpy.linspace(1e13, 1e15, 20),
       'Position' : numpy.zeros((20, 3)),
       'Velocity' : numpy.zeros((20, 3)),
       'vdisp' : numpy.linspace(20, 80, 20),
       'conc' : numpy.ones(20),
       'rvir' : numpy.linspace(0.1, 0.9, 20),
       'Aemit' : numpy.linspace(0.1, 0.9, 20),
    })
        
    crowcanyon.ELG(cat, 1e13, 0.1, 1e13, 1e14, 1, ("Tag", 3))
    crowcanyon.LRG(cat, 1e13, 0.1, 1e13, 1e14, 1, ("Tag", 3))
    crowcanyon.QSO(cat, 1e13, 0.1, ("Tag", 3))

