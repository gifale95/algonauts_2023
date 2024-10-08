"""Fit a linear regression to predict fMRI data using the DNN feature maps as
predictors. The linear regression is trained using the training fMRI data (Y)
and image features (X), and the learned weights are used to synthesize the fMRI
test data using the test image features.

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
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='../algonauts_2023', type=str)
args = parser.parse_args()

print('>>> Algonauts 2023 train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the DNN feature maps
# =============================================================================
data_dir = os.path.join(args.project_dir, 'dnn_feature_maps', 'subj'+
	format(args.sub, '02'))
X_train = np.load(os.path.join(data_dir, 'training_feature_maps.npy'))
X_test = np.load(os.path.join(data_dir, 'test_feature_maps.npy'))


# =============================================================================
# Load the fMRI data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'challenge_data', 'subj'+
	format(args.sub, '02'), 'training_split', 'training_fmri')
y_train_lh = np.load(os.path.join(data_dir, 'lh_training_fmri.npy'))
y_train_rh = np.load(os.path.join(data_dir, 'rh_training_fmri.npy'))


# =============================================================================
# Train the linear regression and save the predicted fMRI for the test images
# =============================================================================
# Empty synthetic fMRI data matrices of shape:
# (Test image conditions × fMRI vertices)
synt_test_lh = np.zeros((X_test.shape[0],y_train_lh.shape[1]))
synt_test_rh = np.zeros((X_test.shape[0],y_train_rh.shape[1]))

# Independently for each fMRI vertex, fit a linear regression using the
# training image conditions and use it to synthesize the fMRI responses of the
# test image conditions
for v in tqdm(range(y_train_lh.shape[1])):
	reg_lh = LinearRegression().fit(X_train, y_train_lh[:,v])
	synt_test_lh[:,v] = reg_lh.predict(X_test)
for v in tqdm(range(y_train_rh.shape[1])):
	reg_rh = LinearRegression().fit(X_train, y_train_rh[:,v])
	synt_test_rh[:,v] = reg_rh.predict(X_test)

# Save the synthetic fMRI test data
save_dir = os.path.join(args.project_dir, 'synthetic_data', 'subj'+
	format(args.sub, '02'))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'lh_test_synthetic_fmri.npy'), synt_test_lh)
np.save(os.path.join(save_dir, 'rh_test_synthetic_fmri.npy'), synt_test_rh)
