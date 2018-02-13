import numpy as np
import numpy.random as rm

import h5py as h5

import os
import datetime
import pdb
import time
import json
from collections import OrderedDict

import scipy.constants as ct
import scipy.integrate as integrate
import scipy.special as spec
from scipy.interpolate import interp1d, interp2d
from scipy.special import logit,expit
from scipy.stats import gaussian_kde
import scipy.stats as stats
from scipy.special import gamma as Gamma_Function
from scipy.integrate import quad
from scipy.special import hyp2f1

from astropy.constants import M_sun
from astropy.cosmology import Planck15 as cosmo
from astropy.units import u
from astropy.io import ascii

from pathos.multiprocessing import Pool
import sys

from mbh_binaries import MassiveBlackHoleBinaries

from sklearn.preprocessing import MinMaxScaler

from astropy.stats import knuth_bin_width

import dill


G=ct.G
c=ct.c
Msun = 1.989e30
pi = ct.pi

h = 0.704


class SimulationRun:

	def __init__(self, kernel_in, sid):
		self.kernel_in, self.sid = kernel_in, sid
		self.now = datetime.datetime.now()

	def prep_interpolations(self):
		#prepare for redshift interpolation in the parallel function
		if 'dcom_ltt_interp.pkl' not in os.listdir(self.sid['input_info']['data_file_location']) or 'z_age_interp.pkl' not in os.listdir(self.sid['input_info']['data_file_location']):
			zs = np.logspace(-4,4, 100000)
			dcom = cosmo.comoving_distance(zs).value*1000 #kpc
			ltt = cosmo.lookback_time(zs).value*1e9 #years
			age = cosmo.age(zs).value*1e9 #years
			self.sid['dcom_ltt_interp'] = interp1d(dcom,ltt)
			with open(self.sid['input_info']['data_file_location'] + '/' + 'dcom_ltt_interp.pkl', 'wb') as f:
				dill.dump(self.sid['dcom_ltt_interp'], f, dill.HIGHEST_PROTOCOL)

			self.sid['z_age_interp'] = interp1d(age, zs)
			with open(self.sid['input_info']['data_file_location'] + '/' + 'z_age_interp.pkl', 'wb') as f:
				dill.dump(self.sid['z_age_interp'], f, dill.HIGHEST_PROTOCOL)

		else:
			with open(self.sid['input_info']['data_file_location'] + '/' + 'dcom_ltt_interp.pkl', 'rb') as f:
				self.sid['dcom_ltt_interp'] = dill.load(f)
			with open(self.sid['input_info']['data_file_location'] + '/' + 'z_age_interp.pkl', 'rb') as f:
				self.sid['z_age_interp'] = dill.load(f)

		return

	def prep_box_coordinates(self):
		#adjust to center of box
		box_side_length = self.sid['mc_generation_info']['box_side_length']*1000 #ckpc

		#figure out number of boxes
		z_lim = 15.0
		side_length = cosmo.comoving_distance(z_lim).value*1000 #kpc

		num_boxes = int(np.ceil((side_length-box_side_length/2)/box_side_length))
		dim1 = np.arange(int(-1*num_boxes),num_boxes+1, 1)

		#grid for 3d
		x, y, z = np.meshgrid(dim1,dim1,dim1)

		#centers of boxes
		box_center_coords = np.transpose(np.array([x.ravel(), y.ravel(), z.ravel()]))*box_side_length #kpc


		#determine which boxes are not in the bounds of the visible universe and remove
		too_far = np.where(np.sqrt(np.sum(box_center_coords**2, axis = 1)) > side_length)[0]
		self.total_boxes = int((2*num_boxes+1)**3)- len(too_far)
		self.box_center_coords = np.delete(box_center_coords, too_far, axis = 0)
		return

	#read in info about parallelism
	def prep_parallelization(self):
		num_splits = self.sid['num_splits']
		self.num_processors = self.sid['num_processors']

		#set up inputs for each processor
		#based on num_splits which indicates max number of boxes per processor
		split_val = int(np.ceil(self.total_boxes/num_splits))

		split_inds = [num_splits*i for i in np.arange(1,split_val)]

		coord_split = np.split(self.box_center_coords,split_inds)


		self.args = []
		for i, bc_split in enumerate(coord_split):
			self.args.append((i, self.kernel_in, bc_split, self.sid))

		return

	def run_parallel_simulation(self):
		print('Total Processes: ',len(self.args))

		#test parallel func
		#check = [parallel_func(*self.args[i]) for i in [100, 1000, 3000, 5000]]
		#pdb.set_trace()

		para_start_time = time.time()
		results = []
		print('numprocs', self.num_processors)
		with Pool(self.num_processors) as pool:
			print('start pool\n')
			results = [pool.apply_async(parallel_func, arg) for arg in self.args]
			out = [r.get() for r in results]

		self.paralllel_sim_duration = time.time() - para_start_time
		#self.sid['needed_values']  = self.sid['needed_values'] + ['Redshift_Coalescence', 'Redshift_Formation', 'Eccentricity_0', 'Dist', 'T_delay', 'Time_Diff']
		self.sid['needed_values'] = self.sid['needed_values'] + ['Time_of_Coalescence','Redshift_Coalescence', 'Redshift_Formation', 'Eccentricity_0', 'Dist', 'Time_Diff']

		self.sid['needed_values_units'] = self.sid['needed_values_units'] + ['Years', 'None', 'None', 'None', 'ckpc', 'Years']

		#if 'Eccentricity_f' not in self.sid['needed_values']:
		#	self.sid['needed_values']  = self.sid['needed_values'] + ['Eccentricity_f']
		#	self.sid['needed_values_units'] = self.sid['needed_values_units'] + ['None']

		
		self.out_parsed = {key: np.concatenate([r[key] for r in out if r != {}]) for key in self.sid['needed_values']}
		return

	def output_simulation_results(self):
		
		sort_inds = np.argsort(self.out_parsed['Time_Diff'])	

		print('\nNumber of boxes with calculation: ', len(self.out_parsed['Time_of_Formation']))

		final_out_ready = {key: self.out_parsed[key][sort_inds] for key in self.out_parsed.keys()}

		num_LISA_detections = len(np.where((final_out_ready['Time_Diff'] <= self.sid['mc_generation_info']['LISA_mission_duration'] + 1.0) & (final_out_ready['Time_Diff']>0.0))[0])

		f_list = os.listdir(self.sid['output_info']['output_file_location'] + '/')
		
		num_list = np.asarray([int(f.split('.')[0].rsplit('_')[-1]) for f in f_list if f[0:len(self.sid['output_info']['out_file_name_start'])] == self.sid['output_info']['out_file_name_start']]).astype(int)
		
		if len(num_list) != 0:
			num_file = num_list.max()+1
		else:
			num_file = 1

		with h5.File(self.sid['output_info']['output_file_location'] + '/' + self.sid['output_info']['out_file_name_start'] + str(num_file) + '.' + self.sid['output_info']['out_file_type'], 'w') as f:
			header = f.create_group('Header')
			
			header.attrs['Title'] = 'Black Hole Merger Monte Carlo Simulation'
			header.attrs['Author'] = 'Michael Katz'
			header.attrs['Date/Time'] = str(self.now)
			header.attrs['Parallel Calculation Time'] = str(self.paralllel_sim_duration) + 'seconds'

			header.attrs['Base Simulation'] = self.sid['Base_Simulation']
			header.attrs['Data Origin File'] = self.sid['input_info']['data_file_name']
			header.attrs['Evolution'] = self.sid['evolve_info']['evolve_func']
			header.attrs['KDE File'] = self.sid['mc_generation_info']['kde_output_file']
			#NEED to ADD Accretion

			header.attrs['Mean Mergers per Box'] = self.sid['mc_generation_info']['mean']
			header.attrs['StdDev of Mergers per Box as Percent of Mean'] = self.sid['mc_generation_info']['std_dev_percent']
		
			header.attrs['Max Merger Time Difference'] = self.sid['mc_generation_info']['max_time_diff']
			header.attrs['Min Merger Time Difference'] = self.sid['mc_generation_info']['min_time_diff']

			header.attrs['Number of Sources'] = len(final_out_ready['Time_of_Formation'])
			header.attrs['Number of Possible LISA Sources'] =  num_LISA_detections
			header.attrs['LISA Mission Duration'] = self.sid['mc_generation_info']['LISA_mission_duration']
			header.attrs['LISA Mission Start'] = self.sid['mc_generation_info']['LISA_mission_start_year']
			
			data = f.create_group('Data')

			print(final_out_ready.keys())
			for i, key in enumerate(self.sid['needed_values']):
				dset = data.create_dataset(key, data = final_out_ready[key], dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)
						
				dset.attrs['unit'] = self.sid['needed_values_units'][i]

		return num_file

def parallel_func(j, kernel_in, bc_coords, sid):
	#need to change coords to center of box coords
	if j%500 == 0:
		print('Process: ', j, 'start', len(bc_coords), 'boxes')

	mc_info_dict = sid['mc_generation_info']
	generated_dict = {}
	#set for correct random generation in parallel
	rm.seed(int(time.time())+j)

	#determine number of boxes to generate coordinates for
	num_boxes = len(bc_coords)

	num_mergers_for_each_box = mc_info_dict['function_for_random_generation'](mc_info_dict['mean'], mc_info_dict['std_dev_percent']*mc_info_dict['mean'], size = num_boxes).astype(int)

	bc_coords_tile = np.repeat(bc_coords, num_mergers_for_each_box,axis=0)

	#print(j, "generate ready")
	
	#generated_binaries = return_from_kernel(kernel_in, len(bc_coords_tile), sid)
	
	generated_binaries = kernel_in.resample(len(bc_coords_tile))
	
	#print(j, "generate finished")

	#should rotate coords at random orientation

	bh_cds_tile = np.transpose(np.asarray([generated_binaries[key] for key in ['Coordinate_x', 'Coordinate_y', 'Coordinate_z']]))

	#change to center of box coordinates
	bh_cds_tile = bh_cds_tile - mc_info_dict['box_side_length']*1000/2.

	new_bh_cds = np.add(bh_cds_tile,bc_coords_tile)

	#find comoving distance to source assuming detector is at (0,0,0)

	#ADD starting coordinate so not default zero
	dist = np.sqrt(np.sum(new_bh_cds**2, axis = 1)) #comoving kpc
	light_travel_time = sid['dcom_ltt_interp'](dist)

	lisa_start = cosmo.age(0.0).value*1e9 + mc_info_dict['LISA_mission_start_year'] - (float(datetime.datetime.now().year) + float(datetime.datetime.now().timetuple().tm_yday)/365.25)

	
	#figure out difference in arrival time of signal to time when LISA is turned on
	generated_binaries['Time_of_Coalescence'] = generated_binaries['Time_of_Formation'] + generated_binaries['T_delay']

	time_diff =  generated_binaries['Time_of_Coalescence'] + light_travel_time - lisa_start

	inds_keep = np.where((time_diff>=mc_info_dict['min_time_diff']) & (time_diff <= mc_info_dict['max_time_diff']))[0]

	if len(inds_keep) == 0:
		return {}

	for i,key in enumerate(['Coordinate_x', 'Coordinate_y', 'Coordinate_z']):
		generated_binaries[key] = new_bh_cds.T[i]

	generated_binaries = {key: generated_binaries[key][inds_keep] for key in generated_binaries.keys()}

	generated_binaries['Redshift_Coalescence'] = sid['z_age_interp'](generated_binaries['Time_of_Coalescence'])
	generated_binaries['Dist'] = dist[inds_keep]
	generated_binaries['Time_Diff'] = time_diff[inds_keep]
	generated_binaries['Eccentricity_0'] = np.full(len(generated_binaries['Time_of_Coalescence']), sid['evolve_info']['eccentricity'])

	generated_binaries['Redshift_Formation'] = sid['z_age_interp'](generated_binaries['Time_of_Formation'])
	
	if j%500 == 0:
		print(j, 'end')

	return generated_binaries


class MonteCarloInputClass:

	def __init__(self, sid):
		self.sid = sid

	def evolve(self, input_dict, evolve_dict):
		self.evolve_dict = evolve_dict
		
		data = ascii.read(input_dict["data_file_location"] + '/' + input_dict["data_file_name"])
		parameters_dict = {key: np.asarray(data[key]) for key in self.sid['needed_values'] + ['Redshift']}

		parameters_dict['Time_of_Formation'] = cosmo.age(parameters_dict['Redshift']).value*1e9
		self.mbh_binary_inputs = MassiveBlackHoleBinaries(parameters_dict, evolve_dict)
		self.mbh_binary_inputs.evolve()
		return


	def adjust_needed_values(self):
		self.sid['needed_values'] = ['Time_of_Formation', 'T_delay'] + self.sid['needed_values']
		self.sid['needed_values_units'] = ['Years', 'Years'] + self.sid['needed_values_units']
		return

	def scale_data(self, min_val=0.0, max_val=1.0):
		self.scaler = MinMaxScaler(feature_range=(min_val, max_val), copy=True)
		self.data = self.scaler.fit_transform(self.data)
		return

	def knuth_bandwidth_determination(self, bw_selection='min'):
		#bandwidth selection is min, max, mean
		bandwidths = np.asarray([knuth_bin_width(data_set) for data_set in self.data.T])
		self.bw = getattr(bandwidths, bw_selection)()
		return

	def make_kernel(self):
		self.data = np.asarray([self.mbh_binary_inputs.par_dict[key] for key in self.sid['needed_values']]).T

		bound = self.sid['mc_generation_info']['bound_on_rescale']
		self.scale_data(0.0+bound, 1.0-bound)		
		self.data = logit(self.data)
		self.knuth_bandwidth_determination()
		getattr(self, self.sid['mc_generation_info']['kde_method'])()
		self.kernel = gaussian_check(self.kernel_class, self.scaler, self.sid['needed_values'])
		return

	def scipy_kde(self):
		self.kernel_class = gaussian_kde(self.data.T, bw_method=self.bw)
		return

	def scikit_learn_kde(self):
		self.kernel_class = KernelDensity(bandwidth=self.bw)
		self.kernel_class.fit(self.data)
		return

	def dill_dump_of_kernel(self):
		with open(self.sid['input_info']['data_file_location'] + '/' + self.sid['mc_generation_info']['kde_output_file'], 'wb') as f:
			dill.dump(self.kernel, f, dill.HIGHEST_PROTOCOL)
		return

class gaussian_check:
	def __init__(self, kernel, scaler, kde_names):
		self.kernel,self.scaler, self.kde_names = kernel, scaler, kde_names

	def resample(self, num):
		self.data_drawn = self.kernel.resample(num).T
		self.data_drawn = expit(self.data_drawn)
		self.data_drawn = self.scaler.inverse_transform(self.data_drawn)
		data_ready = {key:self.data_drawn.T[i] for i,key in enumerate(self.kde_names)}
		return data_ready




def simulation_main(sid):
	random_function_map = {'normal_distribution': rm.normal}

	sid['mc_generation_info']['function_for_random_generation'] = random_function_map[sid['mc_generation_info']['function_for_random_generation']]
	

	monte_carlo_input = MonteCarloInputClass(sid)

	if sid['mc_generation_info']['kde_output_file'] in os.listdir(sid['input_info']['data_file_location'] + '/' ):
		with open(sid['input_info']['data_file_location'] + '/' + sid['mc_generation_info']['kde_output_file'], 'rb') as f:
			print('READ IN KDE')
			monte_carlo_input.kernel = dill.load(f)
			monte_carlo_input.adjust_needed_values()

	else:
		print("make KDE")
		monte_carlo_input.evolve(sid['input_info'], sid['evolve_info'])
		monte_carlo_input.adjust_needed_values()
		monte_carlo_input.make_kernel()
		monte_carlo_input.dill_dump_of_kernel()

	sid_init = monte_carlo_input.sid.copy()

	num_files_out = []
	start_time = time.time()
	for i in np.arange(1,sid_init['num_sims']+1):
		print("\nSimulation %i Started\n" %(i))

		single_sim_start_time = time.time()

		sid = sid_init.copy()
		simulation = SimulationRun(monte_carlo_input.kernel, sid)
		simulation.prep_interpolations()
		simulation.prep_box_coordinates()
		simulation.prep_parallelization()
		simulation.run_parallel_simulation()
		new_file_num = simulation.output_simulation_results()

		num_files_out.append(new_file_num)

		del simulation

		print("\nSimulation %i Finished --- %s seconds ---" % (i, time.time() - start_time))
		print("Simulation %i Duration: %s seconds\n" % (i, time.time() - single_sim_start_time))

	print("%i Simulations Completed with output string %s#.%s"%(sid['num_sims'],sid['output_info']['out_file_name_start'],sid['output_info']['out_file_type']))
	
	print('File Numbers: ', num_files_out)

	return
		
	


if __name__ == '__main__':
	"""
	starter function to read in json and pass to plot_main function. 
	"""
	#read in json
	simulation_info_dict = json.load(open(sys.argv[1], 'r'),
		object_pairs_hook=OrderedDict)
	simulation_main(simulation_info_dict)






