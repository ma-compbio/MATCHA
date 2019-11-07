from pybloomfilter import BloomFilter
import multiprocessing
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
from gensim.models import Word2Vec

import time
import argparse
import warnings
import random
from random_walk import random_walk
from random_walk_hyper import random_walk_hyper
from Modules import *
from utils import *

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.use("Agg")

cpu_num = multiprocessing.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")


def parse_args():
	# Parses the node2vec arguments.
	parser = argparse.ArgumentParser(description="Hyper-SAGNN")

	parser.add_argument('--data', type=str, default='SPRITE')
	parser.add_argument('--TRY', action='store_true')
	parser.add_argument('--FILTER', action='store_true')
	parser.add_argument('--grid', type=str, default='')
	parser.add_argument('--remark', type=str, default='')

	parser.add_argument('--random-walk', action='store_true')

	parser.add_argument('--dimensions', type=int, default=64,
						help='Number of dimensions. Default is 64.')

	parser.add_argument('-l', '--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 40.')

	parser.add_argument('-r', '--num-walks', type=int, default=40,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('-k', '--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--p', type=float, default=2,
						help='Return hyperparameter. Default is 2.')

	parser.add_argument('--q', type=float, default=0.25,
						help='Inout hyperparameter. Default is 0.25.')

	parser.add_argument(
		'-a',
		'--alpha',
		type=float,
		default=0.0,
		help='The weight of random walk -skip-gram loss. Default is ')
	parser.add_argument(
		'--rw',
		type=float,
		default=0.01,
		help='The weight of reconstruction of adjacency matrix loss. Default is ')
	parser.add_argument('-w', '--walk', type=str, default='hyper',
						help='The walk type, empty stands for normal rw')
	parser.add_argument('-d', '--diag', type=str, default='True',
						help='Use the diag mask or not')
	parser.add_argument(
		'-f',
		'--feature',
		type=str,
		default='adj',
		help='Features used in the first step')

	args = parser.parse_args()

	if not args.random_walk:
		args.model_name = 'model_no_randomwalk'
		args.epoch = 25
	else:
		args.model_name = 'model_{}_'.format(args.data)
		args.epoch = 25
	if args.TRY:
		args.model_name = 'try' + args.model_name
		if not args.random_walk:
			args.epoch = 5
		else:
			args.epoch = 1
	# args.epoch = 1
	args.model_name += args.remark
	print (args.model_name)

	args.save_path = os.path.join(
		'../checkpoints/', args.data, args.model_name)
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
	return args


def train_batch_hyperedge(
		model,
		loss_func,
		batch_data,
		batch_weight,
		y=""):
	
	x = batch_data
	w = batch_weight
	
	# When label is not generated, prepare the data
	if len(y) == 0:
		x, y, w, s = generate_negative(x, "train_dict", w)
		x, y, w, s = sync_shuffle([x, y, w, s])
	else:
		s = torch.ones((len(y), 1))

	# forward
	pred, recon_loss = model(x, return_recon=True)
	# , weight=s.float().view(-1, 1).to(device)
	loss = loss_func(pred, y)
	return pred, y, loss, recon_loss, w, s


def train_epoch(
		model,
		loss_func,
		training_data,
		optimizer,
		batch_size):
	# Epoch operation in training phase
	# print (len(train_dict[min_size]), train_dict[min_size].capacity, len(test_dict[min_size]))
	edges, edge_weight = training_data
	y = torch.tensor([])
	# y = training_y
	# Permutate all the data
	if len(y) > 0:
		print ("existing y")
		edges, edge_weight, y = sync_shuffle([edges, edge_weight, y])
	else:
		edges, edge_weight = sync_shuffle([edges, edge_weight])

	model.train()

	bce_total_loss = 0
	recon_total_loss = 0
	acc_list, y_list, pred_list, weight_list, size_list = [], [], [], [], []

	batch_num = int(math.floor(len(edges) / batch_size))
	bar = trange(
		batch_num,
		mininterval=0.1,
		desc='  - (Training) ',
		leave=False,
	)
	for i in bar:
		batch_edge = edges[i * batch_size:(i + 1) * batch_size]
		batch_edge_weight = edge_weight[i * batch_size:(i + 1) * batch_size]
		batch_y = ""
		if len(y) > 0:
			batch_y = y[i * batch_size:(i + 1) * batch_size]
			if len(batch_y) == 0:
				continue

		pred, batch_y, loss_bce, loss_recon, batch_w, batch_s = train_batch_hyperedge(
			model, loss_func, batch_edge, batch_edge_weight, y=batch_y)
		loss = loss_bce * alpha + loss_recon * beta
		# loss = loss_bce + loss_recon

		# acc_list.append(accuracy(pred, batch_y))
		y_list.append(batch_y)
		pred_list.append(pred)
		weight_list.append(batch_w)
		size_list.append(batch_s)

		for opt in optimizer:
			opt.zero_grad()

		# backward
		loss.backward()

		# update parameters
		for opt in optimizer:
			opt.step()

		bar.set_description(" - (Training) BCE:  %.4f  recon: %.4f" %
							(bce_total_loss / (i + 1), recon_total_loss / (i + 1)))
		bce_total_loss += loss_bce.item()
		recon_total_loss += loss_recon.item()
	y = torch.cat(y_list)
	pred = torch.cat(pred_list)
	size_list = torch.cat(size_list)
	weight_list = torch.cat(weight_list)
	auc1_1, auc2_1 = roc_auc_cuda(y, pred, weight_list, max_size)
	acc_1 = accuracy(pred, y, weight_list, max_size)
	
	auc1, auc2 = roc_auc_cuda(y, pred, size_list, max_size)
	acc = accuracy(pred, y, size_list, max_size)
	
	return bce_total_loss / batch_num, recon_total_loss / batch_num, acc_1+acc, auc1_1 + auc1, auc2_1+auc2


def eval_epoch(model, loss_func, validation_data, batch_size):
	''' Epoch operation in evaluation phase '''
	bce_total_loss = 0
	recon_total_loss = 0

	model.eval()
	with torch.no_grad():
		validation_data, validation_weight = validation_data
		y = ""

		validation_data, validation_weight = sync_shuffle(
			[validation_data, validation_weight],10000)

		pred, label, size_list, weight_list = [], [], [], []

		for i in tqdm(range(int(math.floor(len(validation_data) / batch_size))),
					  mininterval=0.1, desc='  - (Validation)   ', leave=False):
			# prepare data
			batch_x = validation_data[i * batch_size:(i + 1) * batch_size]
			batch_w = validation_weight[i * batch_size:(i + 1) * batch_size]

			if len(y) == 0:
				batch_x, batch_y, batch_w, batch_s = generate_negative(
					batch_x, "test_dict", weight=batch_w)
			else:
				batch_y = y[i * batch_size:(i + 1) * batch_size]

			batch_x, batch_y, batch_w, batch_s = sync_shuffle(
				[batch_x, batch_y, batch_w, batch_s])
			pred_batch, recon_loss = model(batch_x, return_recon=True)
			size_list.append(batch_s)
			pred.append(pred_batch)
			label.append(batch_y)
			weight_list.append(batch_w)
			# weight=batch_s.float().view(-1, 1).to(device)
			loss = loss_func(pred_batch, batch_y)
			recon_total_loss += recon_loss.item()
			bce_total_loss += loss.item()

		pred = torch.cat(pred, dim=0)
		label = torch.cat(label, dim=0)
		size_list = torch.cat(size_list, dim=0)
		weight_list = torch.cat(weight_list, dim = 0)
		acc_1 = accuracy(pred, label, weight_list, max_size)
		auc1_1, auc2_1 = roc_auc_cuda(label, pred, weight_list, max_size)
		
		acc = accuracy(pred, label, size_list, max_size)
		auc1, auc2 = roc_auc_cuda(label, pred, size_list, max_size)

	return bce_total_loss / (i + 1), recon_total_loss / \
		(i + 1), acc_1 + acc, auc1_1 + auc1, auc2_1 + auc2


def train(model,
		loss,
		training_data,
		validation_data,
		optimizer,
		epochs,
		batch_size):
	valid_accus = [0]
	# outlier_data = generate_outlier()
	edges, edge_weight = training_data
	training_data_new = training_data
	training_data_generator = DataGenerator(
		edges, edge_weight, int(batch_size), 300, True)

	for epoch_i in range(epochs):

		save_embeddings(model, True)
		print ('[ Epoch', epoch_i, 'of', epochs, ']')

		start = time.time()
		edges_part, edge_weight_part = training_data_generator.next_iter()
		training_data_new = edges_part, edge_weight_part

		bce_loss, recon_loss, train_accu, auc1, auc2 = train_epoch(model, loss, training_data_new, optimizer, batch_size)

		print (
			'  - (Training)   bce: {bce_loss: 7.4f},'
			'recon: {recon_loss: 7.4f}'
			' acc: {accu}, auc: {auc1}, aupr: {auc2}, '
			'elapse: {elapse:3.3f} s'.format(
				bce_loss=bce_loss,
				recon_loss=recon_loss,
				accu=train_accu,
				auc1=auc1,
				auc2=auc2,
				elapse=(
					time.time() - start)))

		start = time.time()
		valid_bce_loss, recon_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(model, loss, validation_data, batch_size)
		print (
			'  - (Validation-hyper) bce: {bce_loss: 7.4f}, recon: {recon_loss: 7.4f},'
			'  acc: {accu},'
			' auc: {auc1}, aupr: {auc2},'
			'elapse: {elapse:3.3f} s'.format(
				bce_loss=valid_bce_loss,
				recon_loss=recon_loss,
				accu=valid_accu,
				auc1=valid_auc1,
				auc2=valid_auc2,
				elapse=(
					time.time() - start)))
		valid_aupr_final = float(valid_auc2.split(" ")[-2])
		valid_accus += [valid_aupr_final]

		checkpoint = {
			'model_link': model.state_dict(),
			'epoch': epoch_i}

		model_name = 'model.chkpt'

		if valid_aupr_final >= max(valid_accus):
			torch.save(checkpoint, os.path.join(args.save_path, model_name))

		torch.cuda.empty_cache()

	checkpoint = torch.load(os.path.join(args.save_path, model_name))
	model.load_state_dict(checkpoint['model_link'])

	valid_bce_loss, recon_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(model, loss, validation_data,
	                                                                            batch_size)
	print(
		'  - (Validation-hyper) bce: {bce_loss: 7.4f}, recon: {recon_loss: 7.4f},'
		'  acc: {accu},'
		' auc: {auc1}, aupr: {auc2},'
		'elapse: {elapse:3.3f} s'.format(
			bce_loss=valid_bce_loss,
			recon_loss=recon_loss,
			accu=valid_accu,
			auc1=valid_auc1,
			auc2=valid_auc2,
			elapse=(
					time.time() - start)))



def neighbor_check(temp, dict):
	return tuple(temp) in dict
	flag = False
	for i in range(len(temp)):
		for j in [-1, 0, 1]:
			a = np.copy(temp)
			a[i] += j
			a.sort()
			if tuple(a) in dict:
				flag = True
				break
		if flag:
			break
	return flag


def generate_negative(x, dict1, weight=""):
	if len(weight) == 0:
		weight = torch.ones(len(x), dtype=torch.float)
	mode = ""
	if dict1 == 'train_dict':
		dict1 = train_dict
		mode = "train"
	elif dict1 == 'test_dict':
		dict1 = test_dict
		mode = "test"

	change_num_list = [[] for i in range(max_size + 1)]
	for s in range(min_size, max_size + 1):
		change_num = np.random.binomial(s, 0.5, len(x) * (neg_num * 2))
		change_num = change_num[change_num != 0]

		change_num_list[s] = list(change_num)

	neg_list = []
	new_x = []
	new_index = []
	neg_weight = []
	max_id = int(num[-1])
	size_list = []
	size_neg_list = []


	for j, sample in enumerate(x):
		for i in range(neg_num):
			# generate decomposed sample
			# if len(sample) > min_size:
			# 	decompose_sample = np.copy(sample)
			# 	decompose_size = int(
			# 		min(max_size - min_size + 1, len(sample) - min_size + 1) * random.random()) + min_size
			# 	if decompose_size == len(sample):
			# 		decompose_sample = np.copy(sample)
			# 	else:
			# 		decompose_sample = np.copy(sample)
			# 		np.random.shuffle(decompose_sample)
			# 		decompose_sample = decompose_sample[:decompose_size]
			# 		decompose_sample.sort()
			#
			# 	if tuple(decompose_sample) not in dict1[len(decompose_sample)]:
			# 		dict1[len(decompose_sample)].add(tuple(decompose_sample))
			# 	if mode == 'train':
			# 		test_dict[len(decompose_sample)].add(tuple(decompose_sample))
			#
			# else:
			# 	decompose_sample = np.copy(sample)
			#
			# 	if tuple(decompose_sample) not in dict1[len(decompose_sample)]:
			# 		dict1[len(decompose_sample)].add(tuple(decompose_sample))
			# 	if mode == 'train':
			# 		test_dict[len(decompose_sample)].add(tuple(decompose_sample))

			decompose_sample = np.copy(sample)
			list1 = change_num_list[decompose_sample.shape[-1]]
			change_num = list1.pop()
			changes = np.random.choice(np.arange(decompose_sample.shape[-1]),change_num,replace=False)
			simple_or_hard = np.random.rand()
			temp = np.copy(decompose_sample)
			trial = 0
			flag = False
			while neighbor_check(temp, dict1[(len(temp))]):
				temp = np.copy(decompose_sample)
				trial += 1
				if trial >= 1000:
					temp = ""
					break
					
				for change in changes:
					if temp[change] not in node2chrom:
						print (temp, decompose_sample)
					chrom = node2chrom[temp[change]]
					start, end = chrom_range[chrom]
					
					# Only change one node
					if simple_or_hard <= pair_ratio:
						# temp[change] = np.random.randint(int(start), int(end), 1)
						temp[change] = int(
							math.floor(
								(end - start) * random.random())) + start
					else:
						# Only one node type
						temp = np.random.randint(
							1, max_id, decompose_sample.shape[-1])
					
				temp = list(set(temp))
				
				if len(temp) < len(decompose_sample):
					temp = np.copy(decompose_sample)
				
				temp.sort()
				dis_list = []
				for k in range(len(temp) - 1):
					dis_list.append(temp[k + 1] - temp[k])
				if np.min(dis_list) <= 5:
					temp = np.copy(decompose_sample)
			
			if len(temp) > 0:
				if i == 0:
					new_x.append(decompose_sample)
					new_index.append(j)
					size_list.append(len(decompose_sample))

				neg_list.append(temp)
				size_neg_list.append(len(temp))
				neg_weight.append(weight[j])
	

	new_weight = weight[np.array(new_index)]
	new_weight = torch.tensor(new_weight)#.to(device)
	neg_weight = torch.tensor(neg_weight)
	size_list = torch.Tensor(np.concatenate(
		[np.array(size_list), np.array(size_neg_list)], axis=0))
	x = np2tensor_hyper(new_x, dtype=torch.long)
	neg = np2tensor_hyper(neg_list, dtype=torch.long)
	x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
	neg = pad_sequence(neg, batch_first=True, padding_value=0).to(device)

	a = torch.cat([x, neg])
	
	return a,\
		torch.cat([torch.ones((len(x), 1), device=device), (torch.zeros((len(neg), 1), device=device))]),\
		torch.cat([new_weight, neg_weight], dim=0),\
		size_list


def save_embeddings(model, origin=False):
	model.eval()
	with torch.no_grad():
		ids = np.arange(num_list[-1]) + 1
		ids = torch.Tensor(ids).long().to(device).view(-1, 1)
		embeddings = []
		for j in range(math.ceil(len(ids) / batch_size)):
			x = ids[j * batch_size:min((j + 1) * batch_size, len(ids))]
			if origin:
				embed = model.get_node_embeddings(x)
			else:
				embed = model.get_embedding_static(x)
			embed = embed.detach().cpu().numpy()
			embeddings.append(embed)

		embeddings = np.concatenate(embeddings, axis=0)[:, 0, :]

		np.save("../mymodel_%d.npy" % (0), embeddings)
		
		if origin:
			old_static = np.load("../mymodel_%d_origin.npy" % (0))
			try:
				update_rate = np.sum((old_static - embeddings) ** 2, axis=-1) / np.sum(old_static ** 2, axis=-1)
				print("update_rate: %f\t%f" % (np.min(update_rate), np.max(update_rate)))
			except:
				pass
			np.save("../mymodel_%d_origin.npy" % (0), embeddings)
			
	torch.cuda.empty_cache()
	return embeddings

def oe(matrix):
	for i in range(len(matrix)):
		if i == 0:
			continue
		x = []
		y = []
		for j in range(len(matrix) - i):
			x.append(j)
			y.append(j + i)
		x = np.array(x)
		y = np.array(y)
		# print (x,y)

		matrix[x , y] /= (np.mean(matrix[x,y])+1e-15)
		matrix[y, x] /= (np.mean(matrix[y,x])+1e-15)
		matrix = np.log(1+matrix)
	return matrix

def predict(model, input):
	model.eval()
	output = []
	with torch.no_grad():
		for j in trange(math.ceil(len(input) / batch_size)):
			x = input[j * batch_size:min((j + 1) * batch_size, len(input))]
			x = np2tensor_hyper(x, dtype=torch.long)
			x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
			output.append(model(x).detach().cpu().numpy())
	output = np.concatenate(output, axis=0)
	torch.cuda.empty_cache()
	return output


args = parse_args()
neg_num = 3
batch_size = 96
neg_num_w2v = 5
bottle_neck = args.dimensions
pair_ratio = 1.0
dynamic_dict = False
max_size = 3
min_size = 3
loss = F.binary_cross_entropy

neighbor_mask = []


chrom_range = np.load("../data/SPRITE/chrom_range.npy")
node2chrom = np.load("../data/SPRITE/node2chrom.npy", allow_pickle=True).item()
num = []
for v in chrom_range:
	num.append(v[1] - v[0])

num_list = np.cumsum(num)
zero_num_list = np.array([0] + list(num_list))
print ("Node type num", num)

data_list = []
for size in range(min_size, max_size + 1):
	for thresh in [[3,5],[5,8],[8,12],"upper"]:
		if type(thresh) == list:
			name = "%d_%d" % (thresh[0],thresh[1])
		else:
			name = thresh
		
		data = np.load("../data/SPRITE/tuples/%s_filter_%d.npy" %(name,size)).astype('int')
		for datum in data:
			data_list.append(datum)

data = np.array(data_list)
print (len(data))

attribute_dict = None

if args.feature == 'adj':
	embeddings_initial = []
	inter_initial = np.load("../data/SPRITE/inter_adj_SPRITE.npy").astype('float32')
	
	adj = np.load("../data/SPRITE/intra_adj_SPRITE.npy").astype('float32')
	# adj = np.load("../data/SPRITE/adj.npy").astype('float32')
	for v in chrom_range:
		# adj_list = []
		# for i in range(5):
		# 	adj = np.load("../data/SPRITE/adj_%d.npy" % i).astype('float32')
		# 	adj = adj[v[0] - 1 :v[1] - 1, v[0] - 1:v[1] - 1]
		# 	# adj = np.log(1+adj)
		# # 	# adj = oe(adj)
		# 	adj = np.corrcoef(adj).astype('float32')
		# 	adj_list.append(adj)
		# temp = np.concatenate(adj_list, axis=-1)

		temp = adj[v[0] - 1:v[1] - 1,v[0] - 1:v[1] - 1]
		# temp = np.log(1 + temp)
		temp = np.corrcoef(temp).astype('float32')
		temp[np.isnan(temp)] = 0.0
		# temp = oe(temp)
		# temp = np.concatenate(
		# 	[adj[v[0]:v[1], 0: v[0]], adj[v[0]:v[1], v[1]:]], axis=-1)
		# temp /= (np.sum(temp, axis=1, keepdims=True) + 1e-10) * 100
		# temp /= (np.max(temp, axis=0, keepdims=True) + 1e-10)
		embeddings_initial.append(temp)
	# embeddings_initial = [adj]
	
	print(chrom_range)
# print (train_weight)
# print (train_weight, np.min(train_weight), np.max(train_weight))
# train_weight_mean = np.mean(train_weight)
# train_weight = train_weight / train_weight_mean * neg_num
# test_weight = test_weight / train_weight_mean * neg_num

num = torch.as_tensor(num)
num_list = torch.as_tensor(num_list)
print (num, num_list)
print ("walk type", args.walk)

if args.feature == 'walk':
	node_list = np.arange(num_list[-1]).astype('int')
	if args.walk == 'hyper':
		walk_path = random_walk_hyper(args, node_list, data)
	else:
		walk_path = random_walk(args, num, data)
	del node_list

compress = True
# Note that, no matter how many node types are here, make sure the
# hyperedge (N1,N2,N3,...) has id, N1 < N2 < N3...
if not dynamic_dict:
	test_dict = build_hash(data, compress=compress, max_size=max_size,
								 min_size=min_size, fname="test")
	train_dict = test_dict
	# train_dict = build_hash(train_data, compress = compress, max_size=max_size, min_size = min_size, fname="test")
else:
	train_dict = [BloomFilter(1e8, 1e-3) for i in range(max_size + 1)]
	test_dict = [BloomFilter(1e8, 1e-3) for i in range(max_size + 1)]

data = []
intra_inter = []
for size in range(min_size, max_size + 1):
	temp_list = []
	for thresh in [[3, 5], [5, 8], [8, 12], "upper"]:
		if type(thresh) == list:
			if size == 3 and thresh[0] <= 5:
				continue
			if size == 4 and thresh[0] <= 3:
				continue
		
		if type(thresh) == list:
			name = "%d_%d" % (thresh[0], thresh[1])
		else:
			name = thresh
		
		temp = np.load("../data/SPRITE/tuples/%s_filter_%d.npy" % (name, size)).astype('int')
		temp_list.append(temp)
		intra_inter.append(np.load("../data/SPRITE/tuples/%s_%d_intra_inter.npy" % (name, size)))
	temp_list = np.concatenate(temp_list,axis = 0)
	for datum in temp_list:
		data.append(datum)


data = np.array(data)
intra_inter = np.concatenate(intra_inter,axis = 0)

index = np.arange(len(data))
weight = np.ones((len(data)), dtype='float32')
weight = intra_inter
np.random.shuffle(index)
split = int(0.5 * len(index))
train_data = data[index[:split]]
test_data = data[index[split:]]
train_weight = weight[index[:split]]
test_weight = weight[index[split:]]


del data

print ("train data amount", len(train_data))

print ("dict_size", len(train_dict[-1]), len(test_dict[-1]))

if args.feature == 'walk':
	# Note that for this part, the word2vec still takes sentences with
	# words starts at "0"
	if not args.TRY and os.path.exists(
		"../%s_wv_%d_%s.npy" %
			(args.data, args.dimensions, args.walk)):
		A = np.load(
			"../%s_wv_%d_%s.npy" %
			(args.data,
			 args.dimensions,
			 args.walk),
			allow_pickle=True)
	else:
		print ("start loading")
		walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
		start = time.time()
		split_num = 20
		pool = ProcessPoolExecutor(max_workers=split_num)
		process_list = []
		walks = np.array_split(walks, split_num)

		result = []
		print ("Start turning path to strs")
		for walk in walks:
			process_list.append(pool.submit(walkpath2str, walk))

		for p in as_completed(process_list):
			result += p.result()

		pool.shutdown(wait=True)

		walks = result
		print (
			"Finishing Loading and processing %.2f s" %
			(time.time() - start))
		print ("Start Word2vec")
		import multiprocessing

		print ("num cpu cores", multiprocessing.cpu_count())
		w2v = Word2Vec(
			walks,
			size=args.dimensions,
			window=args.window_size,
			min_count=0,
			sg=1,
			iter=1,
			workers=multiprocessing.cpu_count())
		wv = w2v.wv
		A = [wv[str(i)] for i in range(num_list[-1])]
		np.save("../%s_wv_%d_%s.npy" %
				(args.data, args.dimensions, args.walk), A)

		from sklearn.preprocessing import StandardScaler

		A = StandardScaler().fit_transform(A)

	A = np.concatenate(
		(np.zeros((1, A.shape[-1]), dtype='float32'), A), axis=0)
	A = A.astype('float32')
	A = torch.tensor(A).to(device)
	print (A.shape)

	node_embedding = Wrap_Embedding(int(
		num_list[-1] + 1), args.dimensions, scale_grad_by_freq=False, padding_idx=0, sparse=False)
	node_embedding.weight = nn.Parameter(A)

elif args.feature == 'adj':
	flag = False
	# node_embedding = MultipleEmbedding_back(
	# 	embeddings_initial,
	# 	bottle_neck,
	# 	flag,
	# 	num_list).to(device)
	
	node_embedding = MultipleEmbedding(
		embeddings_initial,
		bottle_neck,
		flag,
		num_list, chrom_range, inter_initial).to(device)
	# node_embedding = Wrap_Embedding(int(
	# 	num_list[-1] + 1), args.dimensions, scale_grad_by_freq=False, padding_idx=0, sparse=False)

classifier_model = Classifier(
	n_head=8,
	d_model=args.dimensions,
	d_k=16,
	d_v=16,
	node_embedding=node_embedding,
	diag_mask=args.diag,
	bottle_neck=bottle_neck).to(device)

save_embeddings(classifier_model, True)



summary(classifier_model, (3,))

params_list = list(classifier_model.parameters())

if args.feature == 'adj':
	# optimizer = torch.optim.RMSprop(params_list, lr=1e-3)
	optimizer = torch.optim.AdamW(params_list, lr=1e-3, amsgrad=False)
else:
	optimizer = torch.optim.RMSprop(params_list, lr=1e-3)

model_parameters = filter(lambda p: p.requires_grad, params_list)
params = sum([np.prod(p.size()) for p in model_parameters])
print ("params to be trained", params)
#
alpha = 0.0
beta = 1.0
train(classifier_model,
	  loss=loss,
	  training_data=(train_data, train_weight),
	  validation_data=(test_data, test_weight),
	  optimizer=[optimizer], epochs=5, batch_size=batch_size)

alpha = 1.0
beta = 1.0
train(classifier_model,
	  loss=loss,
	  training_data=(train_data, train_weight),
	  validation_data=(test_data, test_weight),
	  optimizer=[optimizer], epochs=30, batch_size=batch_size)

model_name = 'model.chkpt'
checkpoint = torch.load(os.path.join(args.save_path, model_name))
classifier_model.load_state_dict(checkpoint['model_link'])

save_embeddings(classifier_model,True)