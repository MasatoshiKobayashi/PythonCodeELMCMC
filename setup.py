#language_level=3

#### Command to build #################
# python setup.py build_ext --inplace #
#######################################

from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("ImpurityTrend", sources=["ImpurityTrend.pyx"] )
setup(name="ImpurityTrend", ext_modules=cythonize([ext]))

ext = Extension("ElectronLifetimeTrend", sources=["ElectronLifetimeTrend.pyx"] )
setup(name="ElectronLifetimeTrend", ext_modules=cythonize([ext]))

ext = Extension("FormPars", sources=["FormPars.pyx"] )
setup(name="FormPars", ext_modules=cythonize([ext]))

ext = Extension("Fit_func2", sources=["Fit_func2.pyx"] )
setup(name="Fit_func2", ext_modules=cythonize([ext]))

ext = Extension("LnLike", sources=["LnLike.pyx"] )
setup(name="LnLike", ext_modules=cythonize([ext]))

ext = Extension("HistorianData", sources=["HistorianData.pyx"] )
setup(name="HistorianData", ext_modules=cythonize([ext]))

ext = Extension("MCMC_Tools", sources=["MCMC_Tools.pyx"] )
setup(name="MCMC_Tools", ext_modules=cythonize([ext]))
