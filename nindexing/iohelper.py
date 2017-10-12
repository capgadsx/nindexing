import numpy
import astropy.nddata
from astropy.nddata import NDDataRef
from astropy.wcs import wcs
from astropy.io import fits
from astropy import units, log
from .utils import noise_level, denoise_impl

def load_fits(filepath):
    hdulist = fits.open(filepath)
    hduobject = hdulist[0]
    hduobject.verify("fix")
    bscale = 1.0
    bunit = units.Unit('u.Jy/u.beam')
    bzero = 0.0
    mask = numpy.isnan(hduobject.data)
    if 'BSCALE' in hduobject.header:
        bscale = hduobject.header['BSCALE']
    if 'BZERO' in hduobject.header:
        bzero = hduobject.header['BZERO']
    if 'BUNIT' in hduobject.header:
        unit = hduobject.header['BUNIT'].lower().replace('jy','Jy')
        bunit = units.Unit(unit, format='fits')
    for item in hduobject.header.items():
        if item[0].startswith('PC00'):
            hduobject.header.remove(item[0])
    coordinateSystem = wcs.WCS(hduobject.header)
    if len(hduobject.data.shape) == 4:
        log.info('4D Detected: Assuming RA-DEC-FREQ-STOKES, and dropping STOKES')
        coordinateSystem = coordinateSystem.dropaxis(3)
        hduobject.data = hduobject.data.sum(axis=0)*bscale+bzero
        mask = numpy.logical_and.reduce(mask, axis=0)
    elif len(hduobject.data.shape) == 3:
        log.info('3D Detected: Assuming RA-DEC-FREQ')
        hduobject.data = (hduobject.data*bscale) + bzero
    elif len(hduobject.data.shape) == 2:
        log.info('2D Detected: Assuming RA-DEC')
        hduobject.data = (hduobject.data*bscale) + bzero
    else:
        log.error('Only 2-4D data allowed')
        raise TypeError('Only 2-4D data allowed')
    hdulist.close()
    return astropy.nddata.NDDataRef(hduobject.data, uncertainty=None, mask=mask, wcs=coordinateSystem, meta=hduobject.header, unit=bunit)

def denoise_cube(cube):
	nlevel = noise_level(cube)
        print(nlevel)
	return denoise_impl(cube, threshold=nlevel)
