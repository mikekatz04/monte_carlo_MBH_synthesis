import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from astropy.io import ascii
from astropy.stats import knuth_bin_width

import pickle

import pdb

#model from http://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py

class GenerateSamples:
	def __init__(self, kde, scaler=None,  dim_adjuster=None, original_data=None):
		self.scaler, self.dim_adjuster, self.kde, self.original_data = scaler, dim_adjuster, kde, original_data

	def resample(self, num, random_state=0):
		sampled_data = self.kde.sample(num, random_state=random_state)
		if self.dim_adjuster != None:
			sampled_data = self.dim_adjuster.inverse_transform(sampled_data)
		if self.scaler != None:
			sampled_data = self.scaler.inverse_transform(sampled_data)
		return sampled_data

class OriginalData:
	def __init__(self, data_for_kde, kde_names, full_dataset):
		self.data_for_kde, self.kde_names, self.full_dataset = data_for_kde, kde_names, full_dataset

class CreateKDE:
	def __init__(self, data_file, kde_data_names):
		self.input_data = np.asarray(ascii.read(data_file))
		self.data = np.asarray([self.input_data[key] for key in kde_data_names]).T

		self.original_data = OriginalData(self.data, kde_data_names, self.input_data)

	def scale_data(self, min_val=0.0, max_val=1.0):
		self.scaler = MinMaxScaler(feature_range=(min_val, max_val), copy=True)
		self.data = self.scaler.fit_transform(self.data)
		return

	def lower_dimensionality(self, ndim, whiten=False):
		self.pca = PCA(n_components=ndim, whiten=whiten)
		self.data = self.pca.fit_transform(self.data)
		return

	def sklearn_bandwidth_determination(self, tolerance=1e-4, low_estimate=1e-9, high_estimate=1e3, num_points=10, num_iterations=-1, kernels_to_try=[]):

		iteration_num = 1
		while(True):
			try_array = np.logspace(np.log10(low_estimate), np.log10(high_estimate), num_points)
			params = {}
			params['bandwidth'] = try_array
			if kernels_to_try != []:
				params['kernel'] = kernels_to_try
			grid = GridSearchCV(KernelDensity(), params)
			print('start grid.fit')
			grid.fit(self.data)
			print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
			if kernels_to_try != []:
				print("best kernel: {0}".format(grid.best_estimator_.kernel))

			if num_iterations != -1 and iteration_num==num_iterations:
				self.kde = grid.best_estimator_
				return

			elif iteration_num>2 and num_iterations == -1:
				if np.fabs(grid.best_estimator_.bandwidth-current_estimate)/current_estimate <= tolerance:
					self.kde = grid.best_estimator_
				return

			ind = np.where(try_array == grid.best_estimator_.bandwidth)[0][0]

			current_estimate = grid.best_estimator_.bandwidth
			low_estimate = try_array[ind-1]
			high_estimate = try_array[ind+1]

			print('Iteration Number', iteration_num, 'complete')
			iteration_num += 1

	def knuth_bandwidth_determination(self, bw_selection='min'):
		#bandwidth selection is min, max, mean
		bandwidths = np.asarray([knuth_bin_width(data_set) for data_set in self.data.T])
		bw = getattr(bandwidths, bw_selection)()

		self.kde = KernelDensity(bandwidth=bw)
		return


if __name__ == "__main__":
	# load the data

	creation_of_kde = CreateKDE('Ill1_simulation_input_numerical.txt', ['M1','M2','Redshift', 'Mdot_1', 'Subhalo_Stellar_Mass_1'])
	

	creation_of_kde.scale_data(0.0,1.0)
	creation_of_kde.lower_dimensionality(3)
	#creation_of_kde.sklearn_bandwidth_determination(num_iterations=3)
	creation_of_kde.knuth_bandwidth_determination()

	generator_class = GenerateSamples(creation_of_kde.kde, creation_of_kde.scaler, creation_of_kde.pca, creation_of_kde.original_data)

	with open('generate_class_test.pickle', 'wb') as f:
		pickle.dump(generator_class, f, pickle.HIGHEST_PROTOCOL)

	