import dask
import numpy
import distributed
from skimage.filters import threshold_local
from skimage.measure import label,regionprops
from skimage.morphology import binary_opening, disk
from skimage.segmentation import clear_border
from .iohelper import load_fits, denoise_cube
from .utils import spectra, stack_cube, kernel_smooth, kernel_shift, release_dask_futures, generate_stats_table, check_notebook

class IndexingDask(object):
	valid_fields = ['gms_percentile', 'precision', 'random_state', 'samples', 'scheduler']

	def __init__(self):
		self.gms_percentile = 0.05
		self.precision = 0.01
		self.random_state = None
		self.samples = 1000
		self.scheduler = '127.0.0.1:8786'

	def __getattr__(self, name):
		if name.startswith('__') and name.endswith('__'):
			return super(IndexingDask, self).__getattr__(name)
		if name not in self.valid_fields:
			raise ValueError(name+' is not a valid field')

	def __setattr__(self, name, value):
		if name not in self.valid_fields:
			raise ValueError(name+' is not a valid field')
		super(IndexingDask, self).__setattr__(name, value)

	def run(self, files):
		if check_notebook():
			return self.run_on_notebook(files)
		else:
			return self.run_on_console(files)

	def run_on_console(self, files):
		client = distributed.Client(self.scheduler)
		pipeline = self.create_pipeline(files)
		dask_futures = client.compute(pipeline)
		results = distributed.as_completed(dask_futures, with_results=True)
		tables = []
		for future, result in results:
			tables.append(result)
		release_dask_futures(dask_futures)
		return tables

	def run_on_notebook(self, files):
		client = distributed.Client(self.scheduler)
		pipeline = self.create_pipeline(files)
		dask_futures = client.compute(pipeline)
		return dask_futures

	def get_pipeline(self, files):
		return self.create_pipeline(files)

	def create_pipeline(self, files):
		load = lambda file: load_fits(file)
		denoise = lambda cube: denoise_cube(cube)
		slice_cube = lambda cube: spectra(cube.data, self.samples, self.random_state)
		velocity_stacking = lambda cube, slice: self.velocity_stacking(cube, slice)
		compute_w = lambda images: self.optimal_w(images, self.gms_percentile)
		run_gms = lambda stacked_image, w_value: self.gms(stacked_image, w_value, self.precision)
		create_table = lambda cube, stacked_images, slices, labeled_images, file_name: self.measure_shape(cube, stacked_images, slices, labeled_images, file_name)
		load.__name__ = 'load-fits'
		denoise.__name__ = 'denoise-cube'
		slice_cube.__name__ = 'slice-cube'
		velocity_stacking.__name__ = 'vel-stacking'
		compute_w.__name__ = 'compute-w'
		run_gms.__name__ = 'gms'
		create_table.__name__ = 'create-table'
		denoised_cubes = []
		evaluation = []
		for file in files:
			cube = dask.delayed(load)(file)
			denoised_cube = dask.delayed(denoise)(cube)
			denoised_cubes.append(denoised_cube)
		cube_slices = []
		for cube in denoised_cubes:
			slices = dask.delayed(slice_cube)(cube)
			cube_slices.append(slices)
		stacked_cubes = []
		for i, cube_slice in enumerate(cube_slices):
			stacked_cube = dask.delayed(velocity_stacking)(denoised_cubes[i], cube_slice)
			stacked_cubes.append(stacked_cube)
		w_values = []
		for cube in stacked_cubes:
			w_value = dask.delayed(compute_w)(cube)
			w_values.append(w_value)
		gms_results = []
		for i, stacked_cube in enumerate(stacked_cubes):
			gms_result = dask.delayed(run_gms)(stacked_cube, w_values[i])
			gms_results.append(gms_result)
		tables_info = []
		for i, gms_out in enumerate(gms_results):
			table_data = dask.delayed(create_table)(denoised_cubes[i], stacked_cubes[i], cube_slices[i], gms_out, files[i])
			tables_info.append(table_data)
		return tables_info

	def fits_loader(self, file):
		try:
			return load_fits(file)
		except Exception:
			return None

	def cube_denoiser(self, cube):
		if cube is None:
			return None
		else:
			return denoise_cube(cube)

	def run_spectra(self, cube):
		if cube is None:
			return None
		else:
			return spectra(cube.data, self.samples, self.random_state)

	def velocity_stacking(self, cube, slices):
		if cube is None or slices is None:
			return None
		else:
			delayed_function = lambda cube, slice: self.velocity_stacking_delayed(cube, slice)
			delayed_function.__name__ = 'vel-stacking-delayed'
			vstacked_cubes = []
			results = None
			for slice in slices:
				stacked_cube = dask.delayed(delayed_function)(cube, slice)
				vstacked_cubes.append(stacked_cube)
			with distributed.worker_client() as client:
				vstacked_cubes = client.compute(vstacked_cubes)
				results = client.gather(vstacked_cubes)
			return results

	def velocity_stacking_delayed(self, cube, slice):
		cube = stack_cube(cube, cube_slice=slice)
		cube.data[numpy.isnan(cube.data)] = 0
		return cube

	def optimal_w(self, cubes, percentile):
		if cubes is None:
			return None
		else:
			delayed_function = lambda image: self.optimal_w_delayed(image, percentile)
			delayed_function.__name__ = 'compute-w-delayed'
			w_results = []
			for cube in cubes:
				w_value = dask.delayed(delayed_function)(cube.data)
				w_results.append(w_value)
			with distributed.worker_client() as client:
				w_results = client.compute(w_results)
				w_results = client.gather(w_results)
			return w_results

	def optimal_w_delayed(self, image, p):
		radiusMin = 5
		radiusMax = 40
		inc = 1
		image = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
		dims = image.shape
		rows = dims[0]
		cols = dims[1]
		maxsize = numpy.max([rows, cols])
		imagesize = cols * rows
		radius_thresh = numpy.round(numpy.min([rows, cols]) / 4.)
		unit = numpy.round(maxsize / 100.)
		radiusMin = radiusMin * unit
		radiusMax = radiusMax * unit
		radiusMax = int(numpy.min([radiusMax, radius_thresh]))
		radius = radiusMin
		inc = inc * unit
		bg = numpy.percentile(image, p * 100)
		fg = numpy.percentile(image, (1 - p) * 100)
		min_ov = imagesize
		threshold = lambda image, radius, bg, fg: self.optimal_w_threshold(image, radius, bg, fg)
		min_overall = lambda overalls, minimum, radius: self.gms_min_overall(overalls, minimum, radius)
		threshold.__name__ = 'compute-w-threshold'
		min_overall.__name__ = 'compute-w-overall'
		result = 0
		overalls = []
		with distributed.worker_client() as client:
			image_future = client.scatter(image)
			while radius <= radiusMax:
				x = dask.delayed(threshold)(image_future, radius, bg, fg)
				overalls.append(x)
				radius += inc
			min_overall_radius = dask.delayed(min_overall)(overalls, min_ov, radius)
			min_overall_radius = client.compute(min_overall_radius)
			result = client.gather(min_overall_radius)
		return result

	def optimal_w_threshold(self, image, radius, bg, fg):
		tt = int (radius ** 2)
		if tt % 2 == 0:
			tt += 1
		threshold_value = threshold_local(image, tt, method='mean', offset=0)
		g = image > threshold_value
		overall = self.gms_bg_fg_segmentation(image, g, bg, fg)
		return (overall, radius)

	def gms_bg_fg_segmentation(self, f, g, bg, fg):
		dims = f.shape
		rows = dims[0]
		cols = dims[1]
		fp_result = 0
		fn_result = 0
		for row in range(rows):
			for col in range(cols):
				if g[row][col] == True:
					if (numpy.abs(f[row][col] - bg) < numpy.abs(f[row][col] - fg)):
						fp_result += 1
				if g[row][col] == False:
					if (numpy.abs(f[row][col] - bg) > numpy.abs(f[row][col] - fg)):
						fn_result += 1
		overall = fp_result + fn_result
		return overall

	def gms_min_overall(self, overalls, minimum, radius):
		for overall in overalls:
			if overall[0] < minimum:
				minimum = overall[0]
				radius = overall[1]
		return radius

	def gms(self, stacked_cubes, w_values, gms_p):
		if stacked_cubes is None or w_values is None:
			return None
		else:
			delayed_function = lambda image, w_value, precision: self.gms_delayed(image, w_value, precision)
			delayed_function.__name__ = 'gms-delayed'
			gms_results = []
			for i, cube in enumerate(stacked_cubes):
				x = dask.delayed(delayed_function)(cube.data, w_values[i], gms_p)
				gms_results.append(x)
			with distributed.worker_client() as client:
				gms_results = client.compute(gms_results)
				results = client.gather(gms_results)
			return results

	def gms_delayed(self, image, w_value, precision):
		if len(image.shape) > 2:
			raise ValueError('Only 2D data cubes supported')
		dims = image.shape
		rows = dims[0]
		cols = dims[1]
		size = numpy.min([rows, cols])
		precision = size * precision
		image = image.astype('float64')
		diff = (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image))
		tt = w_value ** 2
		if tt % 2 == 0:
			tt += 1
		threshold_val = threshold_local(diff, int(tt), method='mean', offset=0)
		g = diff > threshold_val
		r = w_value / 2
		rMin = 2 * numpy.round(precision)
		image_list = []
		while r > rMin:
			background = numpy.zeros((rows, cols))
			selem = disk(r)
			sub = binary_opening(g, selem)
			sub = clear_border(sub)
			sub = label(sub)
			fts = regionprops(sub)
			image_list.append(sub)
			if len(fts) > 0:
				for props in fts:
					C_x, C_y = props.centroid
					radius = int(props.equivalent_diameter / 2.)
					kern = 0.01 * numpy.ones((2 * radius, 2 * radius))
					krn = kernel_smooth(x=numpy.ones((2 * radius, 2 * radius)), kern=kern)
					krn = numpy.exp(numpy.exp(krn))
					if numpy.max(krn) > 0:
						krn = (krn - numpy.min(krn)) / (numpy.max(krn) - numpy.min(krn))
						background = kernel_shift(background, krn, C_x, C_y)
			if numpy.max(background) > 0:
				background = (background - numpy.min(background)) / (numpy.max(background) - numpy.min(background))
				diff = diff - background
			diff = (diff - numpy.min(diff)) / (numpy.max(diff) - numpy.min(diff))
			tt = int(r * r)
			if tt % 2 == 0:
				tt += 1
			adaptive_threshold = threshold_local(diff, tt, offset=0, method='mean')
			g = diff > adaptive_threshold
			r = numpy.round(r / 2.)
		return image_list

	def measure_shape(self, cube, stacked_images, slice_list, labeled_images, file_name):
		if cube is None or stacked_images is None or slice_list is None or labeled_images is None:
			return None
		else:
			assert len(stacked_images) == len(slice_list) == len(labeled_images)
			gen_table = lambda stacked_cube, labeled_images, min_freq, max_freq, fname: generate_stats_table(stacked_cube, labeled_images, min_freq, max_freq, fname)
			gen_table.__name__ = 'create-table-delayed'
			result_tables = []
			for i, stacked_image in enumerate(stacked_images):
				min_freq = float(cube.wcs.all_pix2world(0, 0, slice_list[i].start, 1)[2])
				max_freq = float(cube.wcs.all_pix2world(0, 0, slice_list[i].stop, 1)[2])
				table = dask.delayed(gen_table)(stacked_image, labeled_images[i], min_freq, max_freq, file_name)
				result_tables.append(table)
			with distributed.worker_client() as client:
				result_tables = client.compute(result_tables)
				result = client.gather(result_tables)
			if len(result) == 0 or result[0] == None:
				return (file_name, self.evaluate_table(len(result)))
			else:
				for table in result:
					l = len(table)
					eval_res = self.evaluate_table(l)
					if eval_res == 0:
						return result
					else:
						return (file_name, eval_res)

	def evaluate_table(self, length):
		if 1 <= length <= 100:
			return 0
		elif length > 100:
			return self.precision**(2/3)
		else:
			return self.precision**(3/2)
