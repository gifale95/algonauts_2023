"""Computed the noise-ceiling-normalized encoding accuracy for using the
Challenge test split.

Parameters
----------
sub : int
	Used NSD subject.
project_dir_dir : str
	Project directory.

"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import pearsonr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='../algonauts_2023', type=str)
args = parser.parse_args()

print('>>> Algonauts 2023 test encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the biological fMRI test data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'challenge_data', 'subj'+
	format(args.sub, '02'), 'test_split', 'test_fmri')
lh_bio_test = np.load(os.path.join(data_dir, 'lh_test_fmri.npy'))
rh_bio_test = np.load(os.path.join(data_dir, 'rh_test_fmri.npy'))


# =============================================================================
# Load the synthetic fMRI test data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'synthetic_data', 'subj'+
	format(args.sub, '02'))
lh_synt_test = np.load(os.path.join(data_dir, 'lh_test_synthetic_fmri.npy'))
rh_synt_test = np.load(os.path.join(data_dir, 'rh_test_synthetic_fmri.npy'))


# =============================================================================
# Correlate the biological and synthetic fMRI test data
# =============================================================================
# Left hemishpere
lh_correlation = np.zeros(lh_bio_test.shape[1])
for v in tqdm(range(lh_bio_test.shape[1])):
	lh_correlation[v] = pearsonr(lh_bio_test[:,v], lh_synt_test[:,v])[0]

# Right hemishpere
rh_correlation = np.zeros(rh_bio_test.shape[1])
for v in tqdm(range(rh_bio_test.shape[1])):
	rh_correlation[v] = pearsonr(rh_bio_test[:,v], rh_synt_test[:,v])[0]


# =============================================================================
# Load the noise ceiling
# =============================================================================
data_dir = os.path.join(args.project_dir, 'challenge_data', 'subj'+
	format(args.sub, '02'), 'test_split', 'noise_ceiling')
lh_noise_ceiling = np.load(os.path.join(data_dir, 'lh_noise_ceiling.npy'))
rh_noise_ceiling = np.load(os.path.join(data_dir, 'rh_noise_ceiling.npy'))


# =============================================================================
# Compute the noise-ceiling-normalized encoding accuracy
# =============================================================================
# Set negative correlation values to 0, so to keep the noise-normalized
# encoding accuracy positive
lh_correlation[lh_correlation<0] = 0
rh_correlation[rh_correlation<0] = 0

# Square the correlation values into r2 scores
lh_r2 = lh_correlation ** 2
rh_r2 = rh_correlation ** 2

# Add a very small number to noise ceiling values of 0, otherwise the noise-
# normalized encoding accuracy cannot be calculated (division by 0 is not
# possible)
lh_idx_nc_zero = np.argwhere(lh_noise_ceiling == 0)
rh_idx_nc_zero = np.argwhere(rh_noise_ceiling == 0)
lh_noise_ceiling[lh_idx_nc_zero] = 1e-14
rh_noise_ceiling[rh_idx_nc_zero] = 1e-14

# Compute the noise-ceiling-normalized encoding accuracy
lh_noise_normalized_encoding = np.divide(lh_r2, lh_noise_ceiling) * 100
rh_noise_normalized_encoding = np.divide(rh_r2, rh_noise_ceiling) * 100

# Set the noise-normalized encoding accuracy to 100 for those vertices in which
# the correlation is higher than the noise ceiling, to prevent encoding
# accuracy values higher than 100%
lh_noise_normalized_encoding[lh_noise_normalized_encoding>100] = 100
rh_noise_normalized_encoding[rh_noise_normalized_encoding>100] = 100


# =============================================================================
# Save the noise-ceiling-normalized encoding accuracy
# =============================================================================
encoding_accuracy = {
	'lh_r2' : lh_r2,
	'rh_r2' : rh_r2,
	'lh_noise_ceiling' : lh_noise_ceiling,
	'rh_noise_ceiling' : rh_noise_ceiling,
	'lh_noise_normalized_encoding' : lh_noise_normalized_encoding,
	'rh_noise_normalized_encoding' : rh_noise_normalized_encoding,
	'lh_idx_nc_zero' : lh_idx_nc_zero,
	'rh_idx_nc_zero' : rh_idx_nc_zero
	}

save_dir = os.path.join(args.project_dir, 'encoding_accuracy')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy_subj' + format(args.sub, '02')

np.save(os.path.join(save_dir, file_name), encoding_accuracy)
