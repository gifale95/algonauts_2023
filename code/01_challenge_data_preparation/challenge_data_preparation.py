"""Prepare the NSD data (fMRI responses and images) for the Algonauts 2023
challenge.

Parameters
----------
sub : int
	Used NSD subject.
nsd_dir : str
	Directory of the NSD folder.
save_dir : str
	Prepared challenge data saving directory.

"""

import argparse
import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from nsdcode.nsd_mapdata import NSDmapdata # https://github.com/cvnlab/nsdcode
import h5py
from PIL import Image


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset', type=str)
parser.add_argument('--save_dir', default='../algonauts_2023/challenge_data',
	type=str)
args = parser.parse_args()

print('>>> Algonauts 2023 data preparation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the NSD experimental design
# =============================================================================
# Load the experimental design info
nsd_expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata', 'experiments',
	'nsd', 'nsd_expdesign.mat'))
# Subtract 1 since the indices start with 1 (and not 0)
masterordering = nsd_expdesign['masterordering'] - 1
subjectim = nsd_expdesign['subjectim'] - 1

# Completed sessions per subject
if args.sub in (1, 2, 5, 7):
	completed_sessions = 40
elif args.sub in (3, 6):
	completed_sessions = 32
elif args.sub in (4, 8):
	completed_sessions = 30

# Image presentation matrix of the selected subject
images_per_session = 750
test_sessions = 3
num_trials = completed_sessions * images_per_session
all_trials = subjectim[args.sub-1,masterordering[0]][:num_trials]

# Training data trials and image conditions
# As the training data, use the image conditions of the N-3 sessions
train_sessions = completed_sessions - test_sessions
num_train_trials = train_sessions * images_per_session
train_trials = all_trials[:num_train_trials]
train_unique_cond = np.unique(train_trials)

# Test data trials and image conditions
# As the test data, use the image conditions (i) falling completely within the
# 3 withheld sessions (i.e. all trials for a given condition are fully
# contained within the withheld sessions); and (ii) were never presented to any
# other subject in the N-3 training sessions (i.e., we exclude all of the 1000
# image conditions shared across subjects, except for the ones falling
# completely within the withheld sessions of the 4 subjects who completed the
# entire 40-sessions NSD experiment)
num_test_trials = test_sessions * images_per_session
test_trials_all = all_trials[-num_test_trials:]
# Remove the image conditions already presented in the training sessions
idx_1 = np.isin(test_trials_all, train_unique_cond, invert=True)
test_trials = test_trials_all[idx_1]
# Remove the image conditions presented to other subjects, except for the ones
# falling completely within the withheld sessions of the 4 subjects who
# completed the entire 40-sessions NSD experiment
shared_img = subjectim[0,:1000]
if args.sub in (1, 2, 5, 7):
	valid_shared_img = []
	for i in shared_img:
		if len(np.where(test_trials == i)[0]) == 3:
			valid_shared_img.append(i)
	valid_shared_img = np.asarray(valid_shared_img)
	idx_valid_shared = np.where(np.in1d(shared_img, valid_shared_img))[0]
	shared_img = np.delete(shared_img, idx_valid_shared)
idx_2 = np.isin(test_trials, shared_img, invert=True)
test_trials = test_trials[idx_2]
test_unique_cond = np.unique(test_trials)


# =============================================================================
# Prepare the fMRI betas
# =============================================================================
# Load the NSDgeneral ROI indices
lh_roi_map_nsd_general = np.squeeze(nib.load(os.path.join(args.nsd_dir,
	'nsddata/freesurfer/fsaverage/label/lh.nsdgeneral.mgz')).get_fdata())
rh_roi_map_nsd_general = np.squeeze(nib.load(os.path.join(args.nsd_dir,
	'nsddata/freesurfer/fsaverage/label/rh.nsdgeneral.mgz')).get_fdata())
# Load the RSC ROI indices
lh_roi_map_rsc = np.squeeze(nib.load(os.path.join(args.nsd_dir,
	'nsddata/freesurfer/fsaverage/label/lh.nsdgeneralRSC.mgz')).get_fdata())
rh_roi_map_rsc = np.squeeze(nib.load(os.path.join(args.nsd_dir,
	'nsddata/freesurfer/fsaverage/label/rh.nsdgeneralRSC.mgz')).get_fdata())
# Create the union between the NSDgeneral ROI indices and the RSC ROI indices
lh_fsaverage_nsd_general_plus = lh_roi_map_nsd_general + lh_roi_map_rsc
lh_fsaverage_nsd_general_plus = lh_fsaverage_nsd_general_plus.astype(int)
rh_fsaverage_nsd_general_plus = rh_roi_map_nsd_general + rh_roi_map_rsc
rh_fsaverage_nsd_general_plus = rh_fsaverage_nsd_general_plus.astype(int)

# Load the fMRI betas in fsaverage space
betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata', 'subj'+
	format(args.sub, '02'), 'fsaverage', 'betas_fithrf_GLMdenoise_RR')
for s in tqdm(range(completed_sessions)):
	lh_file_name = 'lh.betas_session' + format(s+1, '02') + '.mgh'
	rh_file_name = 'rh.betas_session' + format(s+1, '02') + '.mgh'
	lh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		lh_file_name)).get_fdata()))
	rh_betas_sess = np.transpose(np.squeeze(nib.load(os.path.join(betas_dir,
		rh_file_name)).get_fdata()))
	# Mask the vertices with the NSDgeneralPlus ROI
	lh_betas_sess = lh_betas_sess[:,np.where(lh_fsaverage_nsd_general_plus)[0]]
	rh_betas_sess = rh_betas_sess[:,np.where(rh_fsaverage_nsd_general_plus)[0]]
	# z-score the betas of each vertex within each scan session
	sc_lh = StandardScaler()
	sc_rh = StandardScaler()
	lh_betas_sess = sc_lh.fit_transform(lh_betas_sess)
	rh_betas_sess = sc_rh.fit_transform(rh_betas_sess)
	if s == 0:
		lh_betas = lh_betas_sess
		rh_betas = rh_betas_sess
		if args.sub in [6, 8]:
			lh_idx_nan = np.isnan(lh_betas_sess)
			rh_idx_nan = np.isnan(rh_betas_sess)
	else:
		lh_betas = np.append(lh_betas, lh_betas_sess, 0)
		rh_betas = np.append(rh_betas, rh_betas_sess, 0)
		if args.sub in [6, 8]:
			lh_idx_nan = np.append(lh_idx_nan, np.isnan(lh_betas_sess), 0)
			rh_idx_nan = np.append(rh_idx_nan, np.isnan(rh_betas_sess), 0)
del lh_betas_sess, rh_betas_sess

# Find and remove the vertices with NaNs in at least one trial
if args.sub in [6, 8]:
	# Find the NaN vertices
	lh_nan_tot = np.array(np.sum(lh_idx_nan, 0), dtype=bool)
	rh_nan_tot = np.array(np.sum(rh_idx_nan, 0), dtype=bool)
	# Remove the NaN vertices from the beta values
	lh_betas = np.delete(lh_betas, np.where(lh_nan_tot)[0], 1)
	rh_betas = np.delete(rh_betas, np.where(rh_nan_tot)[0], 1)
	# Remove the NaN vertices from the NSDgeneralPlus fsaverage ROI
	lh_idx = np.zeros(len(lh_fsaverage_nsd_general_plus), dtype=bool)
	lh_idx[np.where(lh_fsaverage_nsd_general_plus)[0]] = lh_nan_tot
	lh_fsaverage_nsd_general_plus[np.where(lh_idx)[0]] = 0
	rh_idx = np.zeros(len(rh_fsaverage_nsd_general_plus), dtype=bool)
	rh_idx[np.where(rh_fsaverage_nsd_general_plus)[0]] = rh_nan_tot
	rh_fsaverage_nsd_general_plus[np.where(rh_idx)[0]] = 0

# Save the NSDgeneralPlus indices in fsaverage space
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'roi_masks')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'lh.all-vertices_fsaverage_space'),
	lh_fsaverage_nsd_general_plus)
np.save(os.path.join(save_dir, 'rh.all-vertices_fsaverage_space'),
	rh_fsaverage_nsd_general_plus)

# Extract the training betas, average across repetitions, and format to shape:
# (Unique training image condition × Vertices)
lh_betas_train = np.zeros((len(train_unique_cond),lh_betas.shape[1]))
rh_betas_train = np.zeros((len(train_unique_cond),rh_betas.shape[1]))
for i, img_cond in enumerate(train_unique_cond):
	idx = np.where(train_trials == img_cond)[0]
	lh_betas_train[i] = np.mean(lh_betas[idx], 0)
	rh_betas_train[i] = np.mean(rh_betas[idx], 0)

# Extract the test betas, average across repetitions, and format to shape:
# (Unique test image condition × Vertices)
lh_betas_test = np.zeros((len(test_unique_cond),lh_betas.shape[1]))
rh_betas_test = np.zeros((len(test_unique_cond),rh_betas.shape[1]))
for i, img_cond in enumerate(test_unique_cond):
	idx = np.where(all_trials == img_cond)[0]
	lh_betas_test[i] = np.mean(lh_betas[idx], 0)
	rh_betas_test[i] = np.mean(rh_betas[idx], 0)
del lh_betas, rh_betas

# Convert the betas from float 64 to float32 to reduce size
lh_betas_train = np.float32(lh_betas_train)
rh_betas_train = np.float32(rh_betas_train)
lh_betas_test = np.float32(lh_betas_test)
rh_betas_test = np.float32(rh_betas_test)

# Save the training betas
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'training_split', 'training_fmri')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'lh_training_fmri.npy'), lh_betas_train)
np.save(os.path.join(save_dir, 'rh_training_fmri.npy'), rh_betas_train)
del lh_betas_train, rh_betas_train

# Save the test betas
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'test_split', 'test_fmri')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'lh_test_fmri.npy'), lh_betas_test)
np.save(os.path.join(save_dir, 'rh_test_fmri.npy'), rh_betas_test)
del lh_betas_test, rh_betas_test


# =============================================================================
# Prepare the noise ceiling
# =============================================================================
# Load the noise ceiling SNR
lh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.sub, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).get_fdata())
rh_ncsnr = np.squeeze(nib.load(os.path.join(args.nsd_dir, 'nsddata_betas',
	'ppdata', 'subj'+format(args.sub, '02'), 'fsaverage',
	'betas_fithrf_GLMdenoise_RR', 'rh.ncsnr.mgh')).get_fdata())
# Index the NSDgeneralPlus vertices
lh_ncsnr = lh_ncsnr[np.where(lh_fsaverage_nsd_general_plus)[0]]
rh_ncsnr = rh_ncsnr[np.where(rh_fsaverage_nsd_general_plus)[0]]

# Calculate the noise ceiling from the noise ceiling SNR taking into account
# the amount of image conditions and repetitions of the test data split
rep_1 = 0
rep_2 = 0
rep_3 = 0
for i in test_unique_cond:
	if len(np.where(test_trials_all == i)[0]) == 1:
		rep_1 += 1
	elif len(np.where(test_trials_all == i)[0]) == 2:
		rep_2 += 1
	elif len(np.where(test_trials_all == i)[0]) == 3:
		rep_3 += 1
norm_term = (rep_3/3 + rep_2/2 + rep_1/1) / (rep_3 + rep_2 + rep_1)
lh_noise_ceiling = (lh_ncsnr ** 2) / ((lh_ncsnr ** 2) + norm_term)
rh_noise_ceiling = (rh_ncsnr ** 2) / ((rh_ncsnr ** 2) + norm_term)

# Save the noise ceiling
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'test_split', 'noise_ceiling')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'lh_noise_ceiling.npy'), lh_noise_ceiling)
np.save(os.path.join(save_dir, 'rh_noise_ceiling.npy'), rh_noise_ceiling)


# =============================================================================
# Prepare the ROI mask indices
# =============================================================================
# Save the mapping between ROI names and ROI mask values
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'roi_masks')
roi_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
	format(args.sub, '02'), 'label')
roi_map_files = ['prf-visualrois.mgz.ctab', 'floc-bodies.mgz.ctab',
	'floc-faces.mgz.ctab', 'floc-places.mgz.ctab', 'floc-words.mgz.ctab',
	'streams.mgz.ctab']
roi_name_maps = []
for r in roi_map_files:
	roi_map = pd.read_csv(os.path.join(roi_dir, r), delimiter=' ',
		header=None, index_col=0)
	roi_map = roi_map.to_dict()[1]
	roi_name_maps.append(roi_map)
	np.save(os.path.join(save_dir, 'mapping_'+r[:-9]), roi_map)

# Map the ROI mask indices from subject native space to fsaverage space, and
# then to the NSDgeneralPlus challenge space
lh_roi_files = ['lh.prf-visualrois.mgz', 'lh.floc-bodies.mgz',
	'lh.floc-faces.mgz', 'lh.floc-places.mgz', 'lh.floc-words.mgz',
	'lh.streams.mgz']
rh_roi_files = ['rh.prf-visualrois.mgz', 'rh.floc-bodies.mgz',
	'rh.floc-faces.mgz', 'rh.floc-places.mgz', 'rh.floc-words.mgz',
	'rh.streams.mgz']
# Initiate NSDmapdata
nsd = NSDmapdata(args.nsd_dir)
lh_original_vertices = []
rh_original_vertices = []
lh_nsdgeneralplus_vertices = []
rh_nsdgeneralplus_vertices = []
roi_names = []
for r1 in range(len(lh_roi_files)):
	# Map the ROI masks from subject native to fsaverage space
	lh_fsaverage_roi = np.squeeze(nsd.fit(args.sub, 'lh.white', 'fsaverage',
		os.path.join(roi_dir, lh_roi_files[r1])))
	rh_fsaverage_roi = np.squeeze(nsd.fit(args.sub, 'rh.white', 'fsaverage',
		os.path.join(roi_dir, rh_roi_files[r1])))
	# Store the amount of vertices falling within each ROI
	for r2 in roi_name_maps[r1].items():
		if r2[0] != 0:
			roi_names.append(r2[1])
			lh_original_vertices.append(len(
				np.where(lh_fsaverage_roi == r2[0])[0]))
			rh_original_vertices.append(len(
				np.where(rh_fsaverage_roi == r2[0])[0]))
	# Zero the vertices falling outside NSDgeneralPlus
	lh_fsaverage_roi[np.where(lh_fsaverage_nsd_general_plus == 0)[0]] = 0
	rh_fsaverage_roi[np.where(rh_fsaverage_nsd_general_plus == 0)[0]] = 0
	lh_fsaverage_roi = lh_fsaverage_roi.astype(int)
	rh_fsaverage_roi = rh_fsaverage_roi.astype(int)
	# Store the amount of vertices falling within each ROI
	for r2 in roi_name_maps[r1].items():
		if r2[0] != 0:
			lh_nsdgeneralplus_vertices.append(len(
				np.where(lh_fsaverage_roi == r2[0])[0]))
			rh_nsdgeneralplus_vertices.append(len(
				np.where(rh_fsaverage_roi == r2[0])[0]))
	# Map the ROI masks from fsaverage to NSDgeralPlus challenge space
	lh_challenge_roi = lh_fsaverage_roi[np.where(
		lh_fsaverage_nsd_general_plus)[0]]
	rh_challenge_roi = rh_fsaverage_roi[np.where(
		rh_fsaverage_nsd_general_plus)[0]]
	# Save the ROI mask indices
	np.save(os.path.join(save_dir, lh_roi_files[r1][:-4]+'_fsaverage_space'),
		lh_fsaverage_roi)
	np.save(os.path.join(save_dir, rh_roi_files[r1][:-4]+'_fsaverage_space'),
		rh_fsaverage_roi)
	np.save(os.path.join(save_dir, lh_roi_files[r1][:-4]+'_challenge_space'),
		lh_challenge_roi)
	np.save(os.path.join(save_dir, rh_roi_files[r1][:-4]+'_challenge_space'),
		rh_challenge_roi)


# =============================================================================
# Prepare the stimuli images
# =============================================================================
# Access the ".hdf5" NSD images file
sf = h5py.File(os.path.join(args.nsd_dir,'nsddata_stimuli', 'stimuli', 'nsd',
	'nsd_stimuli.hdf5'), 'r')
sdataset = sf.get('imgBrick')

# Select and save the training images
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'training_split', 'training_images')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
for i, img_cond in enumerate(tqdm(train_unique_cond)):
	img = Image.fromarray(sdataset[img_cond]).convert("RGB")
	file_name = 'train-' + format(i+1, '04') + '_nsd-' + \
		format(img_cond, '05') + '.png'
	img.save(os.path.join(save_dir, file_name), format="png")

# Select and save the test images
save_dir = os.path.join(args.save_dir, 'subj'+format(args.sub, '02'),
	'test_split', 'test_images')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
for i, img_cond in enumerate(tqdm(test_unique_cond)):
	img = Image.fromarray(sdataset[img_cond]).convert("RGB")
	file_name = 'test-' + format(i+1, '04') + '_nsd-' + \
		format(img_cond, '05') + '.png'
	img.save(os.path.join(save_dir, file_name), format="png")
