import pdb

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
import numpy as np

class MassiveBlackHoleBinaries:

	def __init__(self, par_dict, evolve_dict):
		self.par_dict, self.evolve_dict = par_dict, evolve_dict

		self.par_dict['Eccentricity_0'] = np.full(len(self.par_dict[self.evolve_dict['evolve_func_args'][0]]), self.evolve_dict['eccentricity'])
		if self.evolve_dict['eccentricity'] != 0.0:
			ecc_data = np.genfromtxt(sid['e_interp_file_location'] + '/' + sid['e_interp_file'], names = True)
			inds_ecc = np.where(ecc_data['e0'] == sid['eccentricity'])[0]
			self.evolve_dict['ecc_final_interp_func'] = interp2d(ecc_data['q'][inds_ecc], ecc_data['gamma'][inds_ecc], ecc_data['ef'][inds_ecc], bounds_error=False)

		else:
			self.evolve_dict['ecc_final_interp_func'] = None


	def evolve(self):
		evolve_func = globals()[self.evolve_dict['evolve_func']]

		self.par_dict['T_delay'], self.ef = evolve_func(*(self.par_dict[key] for key in self.evolve_dict['evolve_func_args']))
		return

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
			
		#FIX FIX FIX FIX
		e_f = np.concatenate(e_interp_vectorized(q_split, gamma_split))

	else:
		e_f = e_0

	#print(time.time()-st)

	Hardening_GW_timescale = FD_FA_hardening_timescale(r_infl, M, m, q, e_f)*(q>=1e-3) + 0.0*(q<1e-3)

	#pdb.set_trace()
	return large_scale_decay_time + DF_timescale + Hardening_GW_timescale, e_f

	#return np.asarray([large_scale_decay_time, DF_timescale, Hardening_timescale])
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

	beta = beta_func(ksi, gamma)
	#beta = beta_interp_func(gamma)

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