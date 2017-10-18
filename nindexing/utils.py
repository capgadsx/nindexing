import numpy
import distributed
from .cfunctions import segmentationImpl, differenceImpl, erosionImpl
from astropy.nddata import support_nddata, NDDataRef
from astropy.table import Table
from astropy import units
from skimage.measure import regionprops

def release_dask_futures(futures):
	for future in distributed.client.futures_of(futures):
		future.release()

def check_notebook():
	try:
		get_ipython()
		return True
	except NameError:
		return False

@support_nddata
def noise_level(data, mask=None, unit=None):
	if unit is None:
		return root_mean_square(data, mask)
	else:
		return root_mean_square(data, mask) * unit

def root_mean_square(data, mask=None):
	if mask is not None:
		data = fix_mask(data, mask)
	data_square = data * data
	rms = numpy.sqrt(data_square.sum() / data_square.size)
	return rms

def fix_mask(data, mask):
	if isinstance(data, numpy.ma.MaskedArray) and mask is None:
		return data
	else:
		return numpy.ma.MaskedArray(data, mask)

@support_nddata
def denoise_impl(data, wcs=None, mask=None, meta=None, unit=None, threshold=0.0):
	if isinstance(threshold, units.Quantity):
		threshold = threshold.value
	elms = data > threshold
	newdata = numpy.zeros(data.shape)
	newdata[elms] = data[elms]
	return NDDataRef(newdata, uncertainty=None, mask=mask, wcs=wcs, meta=None, unit=unit)

def spectra(data, samples, random_seed=None):
	if random_seed:
		random = numpy.random.RandomState(random_seed)
	else:
		random = numpy.random
	range_x = range(data.shape[2])
	range_y = range(data.shape[1])
	frec = data.shape[0]
	spectra = numpy.zeros(data.shape[0])
	x = random.choice(range_x, samples, replace=True)
	y = random.choice(range_y, samples, replace=True)
	pixels = data[:, y, x].T
	for pixel in pixels:
		processed_pixel = process_pixel(pixel)
		spectra += processed_pixel
	spectra = process_pixel(spectra)
	slices = []
	min_slice = -1
	max_slice = -1
	for i in range(frec -1):
		if spectra[i] != 0:
			if min_slice == -1:
				min_slice = i
			else:
				if spectra[i+1] == 0:
					max_slice = i+1
					slices.append(slice(min_slice, max_slice))
					min_slice = -1
				else:
					if i == frec -2:
						max_slice = i+1
						slices.append(slice(min_slice, max_slice))
	return slices

def process_pixel(pixels):
	pixels = pixels.astype(numpy.float64)
	acum = numpy.cumsum(pixels)
	diff = differenceImpl(acum)
	boxing = segmentationImpl(diff)
	boxing = erosionImpl(boxing)
	return boxing.reshape(-1) * pixels.reshape(-1)

@support_nddata
def stack_cube(data, wcs=None, uncertainty=None, mask=None, meta=None, unit=None, cube_slice=None):
	subcube = data[cube_slice, :, :]
	stacked = numpy.sum(subcube, axis=0)
	if wcs:
		wcs = wcs.dropaxis(2)
		return NDDataRef(stacked, uncertainty=uncertainty, mask=mask, wcs=wcs, meta=meta, unit=unit)
	else:
		return stacked

def kernel_smooth(x, kern, norm=True):
	width = kern.shape[0]
	pad = int(width / 2.0)
	x_w = x.shape[0]
	x_h = x.shape[1]
	if norm:
		k = kern / numpy.sum(abs(kern))
	else:
		k = kern
	x_pad = numpy.lib.pad(x, ((pad, pad), (pad, pad)), 'constant')
	smoothed = numpy.zeros((width, width))
	for col in range(x_w):
		for row in range(x_h):
			tmp = x_pad[row:(row+width), col:(col+width)]
			smoothed[row][col] = numpy.sum(k * tmp)
	return smoothed

def kernel_shift(back, kernel, x, y):
	rows_back = back.shape[0]
	cols_back = back.shape[1]
	rowsKernel = kernel.shape[0]
	colsKernel = kernel.shape[1]
	rowInit = int(x - rowsKernel / 2)
	colInit = int(y - colsKernel / 2)
	for row in range(rowsKernel):
		for col in range(colsKernel):
			if (rowInit + row < rows_back - 1) and (colInit + col < cols_back - 1):
				back[rowInit + row][colInit + col] = kernel[row][col]
	return back

def get_shape(data, intensity_image, wcs):
	objs_props = []
	fts = regionprops(data, intensity_image=intensity_image)
	for obj in fts:
		matrix = wcs.pixel_scale_matrix
		dpp_x = matrix[0, 0]
		dpp_y = matrix[1, 1]
		centroid = wcs.celestial.all_pix2world(obj.centroid[0], obj.centroid[1], 1)
		centroid_ra = centroid[0]
		centroid_dec = centroid[1]
		major_axis = abs(dpp_x) * obj.major_axis_length
		minor_axis = abs(dpp_y) * obj.minor_axis_length
		area = abs(dpp_x) * obj.area
		eccentricity = obj.eccentricity
		solidity = obj.solidity
		filled = obj.area / obj.filled_area
		objs_props.append((centroid_ra, centroid_dec, major_axis, minor_axis, area,
							eccentricity, solidity, filled, obj.max_intensity, obj.min_intensity,
							obj.mean_intensity))
	return objs_props

def generate_stats_table(cube, labeled_images, min_freq, max_freq):
	if len(labeled_images) == 0:
		return None
	objects = []
	for image in labeled_images:
		obj_props = get_shape(image, cube.data, cube.wcs)
		objects.extend(obj_props)
	if len(objects) == 0:
		return None
	names = ['CentroidRa', 'CentroidDec', 'MajorAxisLength', 'MinorAxisLength',
				'Area', 'Eccentricity', 'Solidity', 'FilledPercentaje', 'MaxIntensity', 'MinIntensity', 'AverageIntensity']
	meta = {'name': 'Object Shapes'}
	meta['minfreq_hz'] = min_freq
	meta['maxfreq_hz'] = max_freq
<<<<<<< HEAD
	return Table(rows=objects, names=names, meta=meta)
=======
	if 'OBJECT' in cube.meta:
		meta['target'] = cube.meta['OBJECT']
	else:
		meta['target'] = 'Undefined'
	table = Table(rows=objects, names=names, meta=meta)
	return table
>>>>>>> 523ddf0b00f8d248df83f078aaa8ff34253a3a8c
