import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import QuantileTransformer

import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def proba2matrix(sample, weight=None, proba=None):
	
	sample_left = sample
	weight_left = weight
	sample_left -= np.min(sample_left)
	size = int(np.max(sample_left) + 1 )
	m = np.zeros((size, size), dtype='float32')
	if weight is not None:
		for i in range(sample_left.shape[-1] - 1):
			for j in range( i +1, sample_left.shape[-1]):
				m[sample_left[: ,i], sample_left[: ,j]] += np.maximum(proba * weight_left, proba)
	
	else:
		for i in range(sample_left.shape[-1] - 1):
			for j in range(i + 1, sample_left.shape[-1]):
				m[sample_left[:, i], sample_left[:, j]] += proba
	
	m = m[1: ,1:]
	m = m + m.T
	
	return m




def generate_pair_wise(chrom_id):
	samples = []
	for i in range(chrom_range[chrom_id ,0] ,chrom_range[chrom_id ,1]):
		for j in range( i +min_dis, chrom_range[chrom_id ,1]):
			samples.append([i ,j])
	
	samples = np.array(samples)
	return samples

def predict(model, input):
	model.eval()
	output = []
	new_batch_size = int(1e5)
	with torch.no_grad():
		for j in range(math.ceil(len(input) / new_batch_size)):
			x = input[j * new_batch_size:min((j + 1) * new_batch_size, len(input))]
			x = np2tensor_hyper(x, dtype=torch.long)
			x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
			output.append(model(x).detach().cpu().numpy())
	output = np.concatenate(output, axis=0)
	torch.cuda.empty_cache()
	return output

config = get_config()
min_dis = config['min_distance']
temp_dir = config['temp_dir']
vmin = -0.0
vmax = 1.0


chrom_range = np.load(os.path.join(temp_dir,"chrom_range.npy"))
classifier_model = torch.load(os.path.join(temp_dir, "model2load"))
print ("device", classifier_model.layer_norm1.weight.device)
device_info = classifier_model.layer_norm1.weight.device
device_info = str(device_info).split(":")[-1]
torch.cuda.set_device(int(device_info))
transformer = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
task_mode = 'class'

origin = np.load(os.path.join(temp_dir, "intra_adj.npy")).astype('float32')
# origin = np.load(os.path.join(temp_dir, "edge_list_adj.npy")).astype('float32')
for i in range(len(chrom_range)):
	pair_wise = generate_pair_wise(i)
	# print (pair_wise.shape, pair_wise)
	proba = predict(classifier_model, pair_wise).reshape((-1))
	if task_mode == 'class':
		proba = torch.sigmoid(torch.from_numpy(proba)).numpy()
	else:
		proba = F.softplus(torch.from_numpy(proba)).numpy()
	print ( np.sum(proba >= 0.5) ,proba.shape)
	
	pair_wise_weight = np.array([origin[e[0 ] -1 ,e[1 ] -1] for e in tqdm(pair_wise)])
	
	my_proba = proba2matrix(pair_wise, None, proba)
	coverage = np.sqrt(np.sum(my_proba, axis=-1))
	my_proba = my_proba / coverage.reshape((-1, 1))
	my_proba = my_proba / coverage.reshape((1, -1))
	
	origin_part = proba2matrix(pair_wise, None, pair_wise_weight)
	my = my_proba * origin_part
	
	gap = np.sum(origin_part, axis=-1) == 0
	my[gap ,:] = 0.0
	my[:, gap] = 0.0
	my_proba[gap ,:] = 0.0
	my_proba[:, gap] = 0.0
	
	my = transformer.fit_transform(my.reshape((-1, 1))).reshape((len(my), -1))
	origin_part = transformer.fit_transform(origin_part.reshape((-1, 1))).reshape((len(origin_part), -1))
	my_proba = transformer.fit_transform(my_proba.reshape((-1, 1))).reshape((len(my), -1))
	
	fig = plt.figure(figsize=(5, 5))
	plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
	mask =None
	# print ("matrix", matrix, np.min(matrix), np.max(matrix))
	ax = sns.heatmap(my, cmap="Reds", square=True, mask=mask ,cbar=False, vmin=vmin, vmax=vmax)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.savefig("../chr%d_my.png" %(i+1), dpi=300)
	plt.close(fig)
	
	fig = plt.figure(figsize=(5, 5))
	plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
	mask =None
	# print ("matrix", matrix, np.min(matrix), np.max(matrix))
	ax = sns.heatmap(my_proba, cmap="Reds", square=True, mask=mask ,cbar=False, vmin=vmin, vmax=vmax)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.savefig("../chr%d_my_proba.png" %(i+1), dpi=300)
	plt.close(fig)
	
	fig = plt.figure(figsize=(5, 5))
	plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
	mask =None
	# print ("matrix", matrix, np.min(matrix), np.max(matrix))
	ax = sns.heatmap(origin_part, cmap="Reds", square=True, mask=mask ,cbar=False, vmin=vmin, vmax=vmax)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	plt.savefig("../chr%d_origin.png" %(i+1), dpi=300)
	plt.close(fig)
	
	
