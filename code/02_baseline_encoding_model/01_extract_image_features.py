"""Extract the training/testing feature maps using a pretrained AlexNet, and
downsample them to 1000 PCA components.

Parameters
----------
sub : int
	Used NSD subject.
project_dir_dir : str
	Project directory.

"""

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='../algonauts_2023', type=str)
args = parser.parse_args()

print('>>> Algonauts 2023 extract image features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)


# =============================================================================
# Select the layers of interest and import the model
# =============================================================================
# Lists of AlexNet convolutional and fully connected layers
conv_layers = ['conv1', 'ReLU1', 'maxpool1', 'conv2', 'ReLU2', 'maxpool2',
	'conv3', 'ReLU3', 'conv4', 'ReLU4', 'conv5', 'ReLU5', 'maxpool5']
fully_connected_layers = ['Dropout6', 'fc6', 'ReLU6', 'Dropout7', 'fc7',
	'ReLU7', 'fc8']

class AlexNet(nn.Module):
	def __init__(self):
		"""Select the desired layers and create the model."""
		super(AlexNet, self).__init__()
		self.select_cov = ['maxpool1', 'maxpool2', 'ReLU3', 'ReLU4', 'maxpool5']
		self.select_fully_connected = ['ReLU6' , 'ReLU7', 'fc8']
		self.feat_list = self.select_cov + self.select_fully_connected
		self.alex_feats = models.alexnet(pretrained=True).features
		self.alex_classifier = models.alexnet(pretrained=True).classifier
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

	def forward(self, x):
		"""Extract the feature maps."""
		features = []
		for name, layer in self.alex_feats._modules.items():
			x = layer(x)
			if conv_layers[int(name)] in self.feat_list:
				features.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		for name, layer in self.alex_classifier._modules.items():
			x = layer(x)
			if fully_connected_layers[int(name)] in self.feat_list:
				features.append(x)
		return features

model = AlexNet()
if torch.cuda.is_available():
	model.cuda()
model.eval()


# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extract the feature maps of (1) training images and (2) test images

# Image directories
img_set_dir = os.path.join(args.project_dir, 'challenge_data')
splits_parent = ['training_split', 'test_split']
splits_child = ['training_images', 'test_images']
fmaps_train = []
fmaps_test = []
for s in range(len(splits_parent)):
	image_list = os.listdir(os.path.join(img_set_dir, 'subj'+
		format(args.sub, '02'), splits_parent[s], splits_child[s]))
	image_list.sort()
	# Extract the feature maps
	for image in tqdm(image_list):
		img = Image.open(os.path.join(img_set_dir, 'subj'+
			format(args.sub, '02'), splits_parent[s], splits_child[s],
			image)).convert('RGB')
		input_img = V(centre_crop(img).unsqueeze(0))
		if torch.cuda.is_available():
			input_img=input_img.cuda()
		x = model.forward(input_img)
		for f, feat in enumerate(x):
			if f == 0:
				img_feats = np.reshape(feat.data.cpu().numpy(), -1)
			else:
				img_feats = np.append(img_feats, np.reshape(
					feat.data.cpu().numpy(), -1))
		if splits_child[s] == 'training_images':
			fmaps_train.append(img_feats)
		elif splits_child[s] == 'test_images':
			fmaps_test.append(img_feats)
fmaps_train = np.asarray(fmaps_train)
fmaps_test = np.asarray(fmaps_test)


# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images
# feature maps are also applied to the test images feature maps

# Standardize the data
sc = StandardScaler()
sc.fit(fmaps_train)
fmaps_train = sc.transform(fmaps_train)

# Apply PCA
pca = PCA(n_components=100, random_state=seed)
pca.fit(fmaps_train)
fmaps_train = pca.transform(fmaps_train)

# Save the downsampled feature maps
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps', 'subj'+
	format(args.sub, '02'))
file_name = 'training_feature_maps.npy'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)
del fmaps_train


# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
# Standardize the data
fmaps_test = sc.transform(fmaps_test)

# Apply PCA
fmaps_test = pca.transform(fmaps_test)

# Save the downsampled feature maps
file_name = 'test_feature_maps.npy'
np.save(os.path.join(save_dir, file_name), fmaps_test)
