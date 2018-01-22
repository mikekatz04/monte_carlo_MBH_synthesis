import numpy as np
import numpy.random as rm

import h5py as h5

import os
import datetime
import pdb
import time

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

from multiprocessing import Pool
import multiprocessing
import sys


def evolve_simple(m1,m2,gas_mass): #CHECK 3D APERTURE OF 3KPC
	#accretion_func = args['accretion_func']
	#accretion_args = args['accretion_args']

	rich = (gas_mass >= m1+m2)
	poor = (gas_mass < m1+m2)

	t_delay = (rich*0.1 + poor*5.0)*1e9 #yr

	#m1,m2 = accretion_func(m1,m2,time_to_coalescence, accretion_args)

	return t_delay#, m1,m2

def T_gx_star_1(Lambda, R_e_m, vel_disp_m, m):
	return 1e9*(0.06 * (2./np.log(Lambda)) * (R_e_m/10.0)**2 * (vel_disp_m/300.) * (1e8/m)) #yr




def T_gx_star_2(Lambda, R_e_m, vel_disp_m, vel_disp_s):
	return 1e9*(0.15 * (2./np.log(Lambda)) * (R_e_m/10.0) * (vel_disp_m/300.)**2 * (100./vel_disp_s)**3) #yr 




def FD_FA_large_scale_orbital_decay_timescale(R_e_m, vel_disp_m, m, vel_disp_s=[]):
	#Equations 54, 56, 57
	Lambda = 2**(3./2.)*(vel_disp_m/vel_disp_s)

	Lambda = np.clip(Lambda, np.exp(2.0), np.exp(6.0))

	out = np.asarray([T_gx_star_1(Lambda, R_e_m, vel_disp_m, m), T_gx_star_2(Lambda, R_e_m, vel_disp_m, vel_disp_s)]).T

	return np.max(out, axis =1)



def alpha_func(ksi, gamma):
	b = gamma - 3./2.
	alpha = (Gamma_Function(gamma + 1)/Gamma_Function(gamma - 1/2)) * (4./3.) * np.pi**(-1/2) * 2.**(b - gamma) * ksi**3 * hyp2f1(3./2., -b, 5./2., ksi**2/2.)

	return alpha




def beta_integrand_func(x, ksi, b):
	return x**2 * (2-x**2)**b * np.log((x + ksi)/(x - ksi))

def beta_integral_func(ksi, b):
	integral, err = quad(beta_integrand_func, ksi, 1.4, args = (ksi, b))

	return integral




def beta_func(ksi, gamma):
	beta_integral_vectorized = np.frompyfunc(beta_integral_func, 2, 1)
	b = gamma - 3./2.
	integral = beta_integral_vectorized(ksi, b).astype(np.float64)

	beta = (Gamma_Function(gamma + 1)/Gamma_Function(gamma - 1/2)) * 4*np.pi**(-1/2) * 2**-gamma * integral
	return beta



def delta_func(ksi, gamma):
	b = gamma - 3./2.
	
	delta = (Gamma_Function(gamma + 1)/Gamma_Function(gamma - 1/2)) * 8*np.pi**(-1/2) * (2**(-gamma-1)/(b+1))*ksi * (0.04**(b+1) - (2 - ksi**2)**(b+1))

	return delta




def T_dot_bare(Lambda, alpha, beta, delta, gamma, chi, M, m, r_infl):

	return 1.5e7 * (6.0*alpha + beta + delta)**-1/((3/2. - gamma) * (3. - gamma)) * (chi**(gamma - 3./2.) - 1) * (M/3e9)**(1/2) * (m/1e8)**-1 * (r_infl/300)**(3/2)

	#in T_dot_bare the coulomb logarith is set to 6.0


def T_dot_gx(Lambda, alpha, beta, delta, gamma, chi, M, vel_disp_s):
	 
	return 1.2e7 * (np.log(Lambda)*alpha + beta + delta)**-1/((3. - gamma)**2) * (chi**(gamma - 3.) - 1) * (M/3e9) * (100/vel_disp_s)**3 #years


def find_a_crit(r_infl, m, M, gamma):
	return r_infl*(m/(2*M))**(1/(3 - gamma)) #pc


def FD_FA_Dynamical_Friction_timescale(gamma, r_infl, m, M, q, vel_disp_m, vel_disp_s):
	Lambda = 2**(3./2.)*(vel_disp_m/vel_disp_s)

	Lambda = np.clip(Lambda, 2.0, 6.0)

	#eq. 51 for critical a, multiplies by r_infl for chi
	#a_crit = find_a_crit(r_infl, m, M, gamma)	
	#chi = a_crit/r_infl

	a_h = find_a_h(M, m, q, vel_disp_m)

	chi = a_h/r_infl
	
	ksi = 1.
	alpha = alpha_func(ksi, gamma)
	#alpha = alpha_interp_func(gamma)

	#beta = beta_func(ksi, gamma)
	beta = beta_interp_func(gamma)

	delta = delta_func(ksi, gamma)
	#delta = delta_interp_func(gamma)

	out = np.asarray([T_dot_bare(Lambda, alpha, beta, delta, gamma, chi, M, m, r_infl), T_dot_gx(Lambda, alpha, beta, delta, gamma, chi, M, vel_disp_s)]).T
	return np.min(out, axis=1)





def mass_ratio_func(M, m):
	up = (M >= m)
	down = (M < m)
	return up* (m/M) + down* (M/m)


def k_func(M, m):
	return 0.6 + 0.1*np.log10((M + m) / 3e9)  # log10???


def p_e_func(e, M, m):
	k = k_func(M,m)
	return (1-e**2)*(k + (1-k) * (1-e**2)**4)

def FD_FA_hardening_timescale(r_infl, M, m, q, e, psi=0.3, phi=0.4):
	#Equations 61-63
	#includes gravitational regime

	#Below timescale is from FD's paper
	#if e ==0:
	#	p_e = 1.0
	#else:
		#p_e = p_e_func(e, M, m)
	#T_h_GW = 1.2e9 * (r_infl/300.)**((10 + 4*psi)/(5 + psi)) * ((M+m)/3e9)**((-5-3*psi)/(5+psi)) * phi**(-4/(5+psi)) * (4*q/(1+q)**2)**((3*psi - 1)/(5 + psi))* p_e #years

	#We decided to use Vasiliev's eq. 25

	f_e = f_e_func(e) * (e !=0.0) + 1.0 * (e==0.0)
	
	T_h_GW = 1.7e8 * (r_infl/30.)**((10 + 4*psi)/(5 + psi)) * ((M+m)/1e8)**((-5-3*psi)/(5+psi)) * phi**(-4/(5+psi)) * (4*q/(1+q)**2)**((3*psi - 1)/(5 + psi))* f_e**((1+psi)/(5+psi)) * 20**psi #years

	return T_h_GW


def find_a_h(M, m, q, vel_disp_m):
	return 36.0 * (q/(1+q)**2) * ((M + m)/3e9) * (vel_disp_m/300)**-2 #pc




def f_e_func(e):
	return (1-e**2)**(7/2)/(1 + (73./24.)*e**2 + (37.0/96.)*e**4)




def find_a_GW(r_infl, M, m, q, vel_disp_fin, gamma, e=0):
	
	a_h = find_a_h(M, m, q, vel_disp_fin)

	f_e = f_e_func(e)

	#RHS refers to RHS of equation 64
	RHS = 55.0 * (r_infl/30)**(5./10.) * ((M+m)/1e8)**(-5/10) * f_e**(1/5.) * (4*q/(1 + q)**2)**(4/5)

	#return a_GW
	return a_h/RHS*(q>=1e-3) + a_h*(q<1e-3)




def e_integral_func(e):
	return (e**(29/19) * (1 + (121./304)*e**2)**(1181./2299.))/(1 - e**2)**(3./2.)




def GW_timescale_func(r_infl, M, m, q, gamma, e=0):

	#from shane's gw guide
	
	a_crit = find_a_crit(r_infl, m, M, gamma) #pc

	#convert a_GW to meters
	a_0 = a_crit *ct.parsec
	#convert masses to meters
	m1 = M *M_sun.value*ct.G/(ct.c**2)
	m2 = m *M_sun.value*ct.G/(ct.c**2)

	beta = 64./5. * m1*m2 * (m1+m2)
	if e == 0:
		tau_circ = a_0**4/(4*beta) # meters
		return tau_circ/(ct.c*ct.Julian_year) #c to meters and julian year to years
	
	c_0 = a_0 * (1 - e**2)/e**(12/19) * (1 + (121/304) * e**2)**(-870./2299.)

	e_integral = quad(e_integral_func, 0.0, e)

	tau_merge = (12./19.)*c_0**4/beta*e_integral

	
	return tau_merge/(ct.c*ct.Julian_year) #c to meters and julian year to years



def e_interp_for_vec(q_arr, gamma_arr):
	if isinstance(q_arr, float) == True or isinstance(q_arr, np.float64) == True or isinstance(q_arr, np.float32) == True:
		return ecc_final_interp_func(q_arr,gamma_arr)

	if len(q_arr) < 2:
		return ecc_final_interp_func(q_arr,gamma_arr)

	return ecc_final_interp_func(q_arr,gamma_arr).diagonal()

def evolve_FD_FA(M, m, vel_disp_m,  vel_disp_s, gamma, r_infl, start_dist, e_0):
	#put all quantities into arrays to determine which is major galaxy and minor galaxy

	q = mass_ratio_func(M, m)

	large_scale_decay_time = FD_FA_large_scale_orbital_decay_timescale(start_dist, vel_disp_m, m, vel_disp_s)


	DF_timescale = FD_FA_Dynamical_Friction_timescale(gamma, r_infl, m, M, q, vel_disp_m, vel_disp_s)


	#st = time.time()
	if len(np.where(e_0 != 0.0)[0]) != 0:
		num_splits = 1e2
		split_val = int(len(M)/num_splits)

		split_inds = np.asarray([num_splits*i for i in np.arange(1,split_val)]).astype(int)

		gamma_split = np.split(gamma,split_inds)
		q_split = np.split(q,split_inds)
			
		e_f = np.concatenate(e_interp_vectorized(q_split, gamma_split))

	else:
		e_f = e_0

	#print(time.time()-st)

	Hardening_GW_timescale = FD_FA_hardening_timescale(r_infl, M, m, q, e_f)*(q>=1e-3) + 0.0*(q<1e-3)

	#pdb.set_trace()
	return large_scale_decay_time + DF_timescale + Hardening_GW_timescale, e_f

	#return np.asarray([large_scale_decay_time, DF_timescale, Hardening_timescale])
	

def numerical_based_evole_FD_FA(m1, m2, vel_disp_1,  vel_disp_2, gamma_1, gamma_2, separation, z, e_0):

	#find index of major and minor
	major_1 = (m1 >= m2)
	major_2 = (m1 < m2)

	#small s denotes secondary,small m is primary
	#major black hole mass
	M = m1*major_1 + m2*major_2
	#minor black hole mass
	m = m1*major_2 + m2*major_1

	vel_disp_m = vel_disp_1*major_1 + vel_disp_2*major_2
	vel_disp_s = vel_disp_1*major_2 + vel_disp_2*major_1

	#find large scale orbital decay time

	gamma = gamma_1*major_1 + gamma_2*major_2

	##r_infl determined analytically
	r_infl = influence_radius(M, z)

	gamma = np.clip(gamma, 0.55, 2.49)

	return evolve_FD_FA(M, m, vel_disp_m,  vel_disp_s, gamma, r_infl, separation, e_0)


def analytical_based_evole_FD_FA(m1, m2, z, gamma_1=0.55, gamma_2=0.55, e=0):

	#find index of major and minor
	major_1 = (m1 >= m2)
	major_2 = (m1 < m2)

	#small s denotes secondary,small m is primary
	#major black hole mass
	M = m1*major_1 + m2*major_2
	#minor black hole mass
	m = m1*major_2 + m2*major_1

	gamma = gamma_1*major_1 + gamma_2*major_2
	
	R_e = half_radius(M,z)

	vel_disp_m = velocity_dispersion(M,z)

	vel_disp_s = velocity_dispersion(m,z)

	r_infl = influence_radius(M,z)

	return evolve_FD_FA(M, m, vel_disp_m,  vel_disp_s, gamma, r_infl, R_e, e=0)

def half_radius(M_p,z):
	#from FD
	#factor_1=1.0
	factor_1=3.0/(3.**(0.73))
	M_p_0=M_p/((1.+z)**(-0.6))
	factor_2=(M_p_0/(1.E+8))**0.66
	
	half_radius=factor_1*factor_2*(1.+z)**(-0.71)
	
	return half_radius

def velocity_dispersion(M_p,z):
 
	factor_1=190.0
	
	M_p_0=M_p/((1.+z)**(-0.6))
	#velocity_dispersion=factor_1*factor_2*(1.+z)**(0.44)
	
	#velocity_dispersion=(M_p/1.E+8/1.66)**(1./5.)*200.  used before
	
	factor_2=(M_p_0/(1.E+8))**0.2
	
	velocity_dispersion=factor_1*factor_2*(1.+z)**(0.056)
	
	return velocity_dispersion

def influence_radius(M_p,z):

	#inf_radius=G*M_p/(velocity_dispersion(M_p,z)**2)
	inf_radius=35.*(M_p/1.E+8)**(0.56)
	#inf_radius=13.*(M_p/1.e8)/(velocity_dispersion(M_p,z)/200.)**2
	
	return inf_radius


def simulation_func(kernel_in, sid):


	now = datetime.datetime.now()

	#prepare for redshift interpolation in the parallel function
	zs = np.logspace(-4,4, 100000)
	dcom = cosmo.comoving_distance(zs).value*1000 #kpc
	ltt = cosmo.lookback_time(zs).value*1e9 #years
	age = cosmo.age(zs).value*1e9 #years
	sid['dcom_ltt_interp'] = interp1d(dcom,ltt)
	sid['z_age_interp'] = interp1d(age, zs)

	#adjust to center of box
	box_side_length = sid['box_side_length']*1000 #ckpc

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
	total_boxes = int((2*num_boxes+1)**3)- len(too_far)
	box_center_coords = np.delete(box_center_coords, too_far, axis = 0)

	#read in info about parallelism
	
	num_splits = sid['num_splits']
	num_processors = sid['num_processors']

	#set up inputs for each processor
	#based on num_splits which indicates max number of boxes per processor
	split_val = int(np.ceil(total_boxes/num_splits))

	split_inds = [num_splits*i for i in np.arange(1,split_val)]

	coord_split = np.split(box_center_coords,split_inds)

	#initialize arguments for parallelism

	gamma_for_interp = np.linspace(0.51, 2.49,10000)

	#sid['alpha_interp_func'] = interp1d(gamma_for_interp, alpha_func(1.0, gamma_for_interp), bounds_error=False, fill_value='extrapolate')

	sid['beta_interp_func'] = interp1d(gamma_for_interp, beta_func(1.0, gamma_for_interp), bounds_error=False, fill_value='extrapolate')

	#sid['delta_interp_func'] = interp1d(gamma_for_interp, delta_func(1.0, gamma_for_interp), bounds_error=False, fill_value='extrapolate')

	if sid['eccentricity'] != 0.0:
		ecc_data = np.genfromtxt(sid['data_file_location'] + '/' + sid['e_interp_file'], names = True)
		inds_ecc = np.where(ecc_data['e0'] == sid['eccentricity'])[0]
		sid['ecc_final_interp_func'] = interp2d(ecc_data['q'][inds_ecc], ecc_data['gamma'][inds_ecc], ecc_data['ef'][inds_ecc], bounds_error=False)

	else:
		sid['ecc_final_interp_func'] = None

	args = []
	for i, bc_split in enumerate(coord_split):
		args.append((i, kernel_in, bc_split, sid))

	print('Total Processes: ',len(args))

	#test parallel func
	#check = parallel_func(*args[3000])
	#pdb.set_trace()

	results = []
	with Pool(num_processors) as pool:
		print('start pool\n')
		results = [pool.apply_async(parallel_func, arg) for arg in args]

		out = [r.get() for r in results]

	out_parsed = np.concatenate([r for r in out if len(r) != 0])
	sort_inds = np.argsort(np.transpose(out_parsed)[-1])	

	print('\nNumber of boxes with calculation: ', len(out_parsed))

	final_out_ready = np.transpose(out_parsed[sort_inds]).astype(np.float64)

	num_LISA_detections = len(np.where((final_out_ready[-1] <= sid['LISA_mission_duration'] + 1.0) & (final_out_ready[-1]>0.0))[0])

	f_list = os.listdir(sid['output_file_location'] + '/')
	
	num_list = np.asarray([int(f.split('.')[0].rsplit('_')[-1]) for f in f_list if f[0:len(sid['out_file_name_start'])] == sid['out_file_name_start']]).astype(int)

	sid['needed_values'][sid['needed_values'].index('Redshift')] = 'Redshift Formation'
	
	if len(num_list) != 0:
		num_file = num_list.max()+1
	else:
		num_file = 1

	with h5.File(sid['output_file_location'] + '/' + sid['out_file_name_start'] + str(num_file) + sid['out_file_name_end'], 'w') as f:
		header = f.create_group('Header')
		
		header.attrs['Title'] = 'Black Hole Merger Monte Carlo Simulation'
		header.attrs['Author'] = 'Michael Katz'
		header.attrs['Date/Time'] = 'Date/Time: ' + str(now)

		header.attrs['Base_Simulation'] = sid['Base_Simulation']
		header.attrs['Data Origin File'] = sid['data_file_name']
		header.attrs['Evolution'] = sid['evolve_func_key']
		#NEED to ADD Accretion

		header.attrs['Mean Mergers per Box'] = sid['number generation']['mean']
		header.attrs['StdDev of Mergers per Box as Percent of Mean'] = sid['number generation']['std_dev_percent']
	
		header.attrs['Max Merger Time Difference'] = sid['max_time_diff']
		header.attrs['Min Merger Time Difference'] = sid['min_time_diff']

		header.attrs['Number of Sources'] = len(out_parsed)
		header.attrs['Number of Possible LISA Sources'] =  num_LISA_detections
		header.attrs['LISA Mission Duration'] = sid['LISA_mission_duration']
		header.attrs['LISA Mission Start'] = sid['LISA_mission_start_year']
		
		data = f.create_group('Data')

		sid['needed_values'] = sid['needed_values'] + ['Eccentricity_f','time_of_coalescence','dist', 'evolution_time', 'Redshift Coalescence', 'time_diff']

		sid['needed_values_units'] = sid['needed_values_units'] + ['None', 'Years', 'ckpc', 'Years', 'None', 'Years']

		for i in range(len(final_out_ready)):
			dset = data.create_dataset(sid['needed_values'][i], data = final_out_ready[i], dtype = 'float64', chunks = True, compression = 'gzip', compression_opts = 9)
					
			dset.attrs['unit'] = sid['needed_values_units'][i]

	return num_file







def parallel_func(j, kernel_in, bc_coords, sid):

	global alpha_interp_func, beta_interp_func, delta_interp_func, ecc_final_interp_func, e_interp_vectorized

	#alpha_interp_func = sid['alpha_interp_func']
	beta_interp_func = sid['beta_interp_func']
	#delta_interp_func = sid['delta_interp_func']
	ecc_final_interp_func = sid['ecc_final_interp_func']

	e_interp_vectorized = np.frompyfunc(e_interp_for_vec, 2, 1)
	
	#need to change coords to center of box coords
	if j%100 == 0:
		print('Process: ', j, 'start', len(bc_coords), 'boxes')
	#set for correct random generation in parallel
	rm.seed(int(time.time())+j)

	#determine number of boxes to generate coordinates for
	num_boxes = len(bc_coords)

	num_mergers_for_each_box = sid['number generation']['function_for_num'](sid['number generation']['mean'], sid['number generation']['std_dev_percent']*sid['number generation']['mean'], size = num_boxes).astype(int)

	bc_coords_tile = np.repeat(bc_coords, num_mergers_for_each_box,axis=0)

	#print(j, "generate ready")
	generated_binaries = return_from_kernel(kernel_in, len(bc_coords_tile), sid['data_mins'], sid['data_maxs'])
	#print(j, "generate finished")
	#should rotate coords at random orientation

	bh_cds_tile = np.transpose(np.asarray([generated_binaries[sid['needed_values'].index(key)] for key in ['Coordinate_x', 'Coordinate_y', 'Coordinate_z']]))

	#change to center of box coordinates
	bh_cds_tile = bh_cds_tile - sid['box_side_length']*1000/2.

	
	new_bh_cds = np.add(bh_cds_tile,bc_coords_tile)

	#find comoving distance to source assuming detector is at (0,0,0)
	dist = np.sqrt(np.sum(new_bh_cds**2, axis = 1)) #comoving kpc
	light_travel_time = sid['dcom_ltt_interp'](dist)


	#print(j, "ltt calc")

	time_limit = cosmo.age(0.0).value*1e9 + sid['max_time_diff'] + sid['LISA_mission_start_year'] - (float(datetime.datetime.now().year) + float(datetime.datetime.now().timetuple().tm_yday)/365.25)
	
	keep = np.where(generated_binaries[sid['needed_values'].index('time_of_merger')] + light_travel_time <= time_limit)[0]

	if len(keep) == 0:
		return np.array([])

	generated_binaries = generated_binaries[:,keep]
	light_travel_time = light_travel_time[keep]             
	dist = dist[keep]
	new_bh_cds = new_bh_cds[keep,:]

	#add eccentricity column to generated_binaries --> right now it is constant ecc
	generated_binaries = np.append(generated_binaries, np.array([np.full(len(dist), sid['eccentricity'])]), axis=0)
	
	#print(j, "pre_evolve")
	t_delay, e_f = sid['evolve_func'](*(generated_binaries[sid['needed_values'].index(key)] for key in sid['evolve_func_args']))
	time_of_coalescence = generated_binaries[sid['needed_values'].index('time_of_merger')] + t_delay

	#print(j, "post_evolve")
	
	#figure out difference in arrival time of signal to time when LISA is turned on
	time_diff =  time_of_coalescence + light_travel_time - (time_limit - sid['max_time_diff'])

	inds_keep = np.where((time_diff>=sid['min_time_diff']) & (time_diff <= sid['max_time_diff']))[0]

	if len(inds_keep) == 0:
		return np.array([])

	Redshift_coalescence = sid['z_age_interp'](time_of_coalescence[inds_keep].astype(np.float64))

	for i,key in enumerate(['Coordinate_x', 'Coordinate_y', 'Coordinate_z']):
		generated_binaries[sid['needed_values'].index(key)] = new_bh_cds.T[i]

	out_info = np.transpose(np.append(generated_binaries[:,inds_keep], np.array([e_f[inds_keep], time_of_coalescence[inds_keep], dist[inds_keep], t_delay[inds_keep], Redshift_coalescence, time_diff[inds_keep]]), axis = 0))


	if j%100 == 0:
		print(j, 'end')

	return out_info

		





def get_large_scale_data_func(file_name, file_location, needed_values, skip_lines, delimiter):
	
	ill_data = np.genfromtxt(file_location + '/' + file_name, delimiter = delimiter, names = True, skip_header=skip_lines)

	#Change based on data input when ready with illustris1
	ill_data_dict = {key:ill_data[key] for key in ill_data.dtype.names}
	ill_data_dict['time_of_merger'] = cosmo.age(ill_data['Redshift']).value*1e9 #years

	data_for_kernel = np.vstack([ill_data_dict[key] for key in needed_values])
	rid = np.where(data_for_kernel == 0.0)[1]
	
	data_for_kernel = np.delete(data_for_kernel, rid, axis = 1)
	
	return data_for_kernel

def return_from_kernel(kernel, num_gen, data_mins, data_maxs):
	data = kernel.resample(num_gen)
	
	output_data = expit(data)

	scale_up = np.asarray([dat*data_maxs[i] for i,dat in enumerate(output_data)])

	final_out = np.asarray([dat+data_mins[i] for i,dat in enumerate(scale_up)])

	return final_out

def prepare_for_kernel(data_input):

	data_mins = np.asarray([dat.min()*(1+ ((dat.min()<0.0)*1. + (dat.min()>=0.0)*-1.)*1e-6) for dat in data_input])

	zeroed_data = np.asarray([dat-data_mins[i] for i,dat in enumerate(data_input)])
	
	data_maxs = np.asarray([dat.max()*(1+1e-6) for dat in zeroed_data])

	normed_data = np.asarray([dat/data_maxs[i] for i,dat in enumerate(zeroed_data)])

	kernel_input = logit(normed_data)

	return kernel_input, data_mins, data_maxs


def make_kernel(data_input):

	kernel_input, data_mins, data_maxs = prepare_for_kernel(data_input)
	
	kernel = gaussian_kde(kernel_input)

	return kernel, data_mins, data_maxs

def guide_func(sid):
	data_for_kernel = get_large_scale_data_func(sid['data_file_name'],sid['data_file_location'] ,sid['needed_values'], sid['skip_lines'], sid['delimiter'])

	binary_kernel, sid['data_mins'], sid['data_maxs'] = make_kernel(data_for_kernel)

	sid['needed_values'] = sid['needed_values'] + ['Eccentricity_0']
	sid['needed_values_units'] = sid['needed_values_units'] + ['None']

	new_file_num = simulation_func(binary_kernel, sid)
	return new_file_num


def simulation_main(sid):
	#Define Constants 
	
	G=ct.G
	c=ct.c
	Msun = 1.989e30
	pi = ct.pi

	h = 0.704

	evolve_func_map = {'Evolve_Simple': evolve_simple, 'Evolve_numerical_FD': numerical_based_evole_FD_FA, 'Evolve_analytical_FD': analytical_based_evole_FD_FA}

	random_function_map = {'normal_distribution': rm.normal}

	delimiter_map = {'comma': ',', 'tab': '\t'}

	#method_dict = input_dict
	sid['evolve_func'] = evolve_func_map[sid['evolve_func_key']]

	#prepare number generation information
	sid['number generation'] = {'mean':float(sid['mean']),'std_dev_percent': float(sid['std_dev_percent']), 'function_for_num': random_function_map[sid['function_for_num']]}

	sid['delimiter'] = delimiter_map[sid['delimiter']]


	sid['box_side_length'] = float(sid['box_side_length']) #mpc
	sid['max_time_diff'] = float(sid['max_time_diff']) #years
	sid['min_time_diff'] = float(sid['min_time_diff']) #years

	sid['eccentricity'] = float(sid['eccentricity'])

	sid['LISA_mission_duration'] = float(sid['LISA_mission_duration']) #years
	sid['LISA_mission_start_year'] = float(sid['LISA_mission_start_year']) #year

	sid['skip_lines'] = int(sid['skip_lines'])

	sid['num_processors'] = int(sid['num_processors'])
	sid['num_splits'] = int(sid['num_splits'])
	sid['num_sims'] = int(sid['num_sims'])

	sid_init = sid.copy()
	
	num_files_out = []
	start_time = time.time()
	for i in range(sid_init['num_sims']):
		print("\nSimulation %i Started\n" %(i+1))

		single_sim_start_time = time.time()

		sid = sid_init.copy()
		num_file_new = guide_func(sid)

		num_files_out.append(num_file_new)

		print("\nSimulation %i Finished --- %s seconds ---" % (i+1, time.time() - start_time))
		print("Simulation %i Duration: %s seconds\n" % (i+1, time.time() - single_sim_start_time))

	print("%i Simulations Completed with output string %s#%s"%(sid['num_sims'],sid['out_file_name_start'],sid['out_file_name_end']))
	
	print('File Numbers: ', num_files_out)
		
	


if __name__ == '__main__':

	simulation_main(sid)









