from pybloom_live import BloomFilter
import multiprocessing
from torch.nn.utils.rnn import pad_sequence

import time
import argparse
import warnings
import random
from Modules import *
from utils import *


import datetime


cpu_num = multiprocessing.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

warnings.filterwarnings("ignore")
def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		id = int(np.argmax(memory_available))
		print("setting to gpu:%d" % id)
		torch.cuda.set_device(id)
	else:
		return

if torch.cuda.is_available():
	get_free_gpu()

def forward_op_batch(
		model,
		loss_func,
		batch_data,
		batch_weight,
		y=""):
	x = batch_data
	w = batch_weight

	# When label is not generated, prepare the data
	if len(y) == 0:
		x, y, w, s = generate_negative(x, "train_dict", w, neg_num=neg_num)
		x, y, w, s = sync_shuffle([x, y, w, s])
	else:
		s = torch.ones((len(y), 1))

	# forward
	pred, recon_loss = model(x, return_recon=True)
	
	loss = loss_func(pred, y, weight=w)

	return F.sigmoid(pred), y, loss, recon_loss, w, s

def forward_op_batch_regress(
		model,
		loss_func,
		batch_data,
		batch_weight,
		y=""):
	x = batch_data
	w = batch_weight

	# When label is not generated, prepare the data
	if len(y) == 0:
		x, y, w, s = generate_negative(x, "train_dict", w,neg_num=1)
		x, y, w, s = sync_shuffle([x, y, w, s])
	else:
		s = torch.ones((len(y), 1))

	if len(x) % 2 == 1:
		batch_length = (len(x) - 1)
		x = x[:batch_length]
		w = w[:batch_length]
		y = y[:batch_length]
		s = s[:batch_length]
		batch_length = int(batch_length / 2)
	else:
		batch_length = int(len(x) / 2)

	# forward
	pred, recon_loss = model(x, return_recon=True)
	pred = F.softplus(pred)
	
	loss = F.mse_loss(pred, y)
	
	pred = pred.view(batch_length,2)
	y = y.view(batch_length,2)
	w = w.view(batch_length,2)
	s = s.view(batch_length,2)
	l_back = torch.argmin(y,dim=-1,keepdim=False)
	l = l_back.clone()

	l[l == 0] = -1
	mask = y[:,0]!=y[:,1]
	l = l[mask].float()
	l_back = l_back[mask].float()
	pred = pred[mask]
	w = w[mask,0]
	s = s[mask,0]
	# print ("l,pred",l, pred)
	# , weight=s.float().view(-1, 1).to(device)
	# loss = loss_func(pred[:,0], pred[:,1], l, margin=0.1)

	y = l_back
	pred = pred[:,0] - pred[:,1]
	# pred = F.sigmoid(pred)
	# loss = loss_func(pred, y)
	# print (y)
	return F.sigmoid(pred), y, loss, recon_loss, w, s



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
		print("existing y")
		edges, edge_weight, y = sync_shuffle([edges, edge_weight, y])
	else:
		edges, edge_weight = sync_shuffle([edges, edge_weight])

	model.train()

	if task_mode == 'class':
		forward_func = forward_op_batch
	elif task_mode == 'regress':
		forward_func = forward_op_batch_regress

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

		pred, batch_y, loss_bce, loss_recon, batch_w, batch_s = forward_func(
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

	auc1, auc2 = roc_auc_cuda(y, pred, size_list, max_size)
	acc = accuracy(pred, y, size_list, max_size)

	return bce_total_loss / batch_num, recon_total_loss / batch_num, acc, auc1, auc2


def eval_epoch(model, loss_func, validation_data, batch_size):
	''' Epoch operation in evaluation phase '''
	bce_total_loss = 0
	recon_total_loss = 0

	model.eval()

	if task_mode == 'class':
		forward_func = forward_op_batch
	elif task_mode == 'regress':
		forward_func = forward_op_batch_regress

	with torch.no_grad():
		validation_data, validation_weight = validation_data
		y = ""

		validation_data, validation_weight = sync_shuffle(
			[validation_data, validation_weight], 10000)

		pred, label, size_list, weight_list = [], [], [], []

		for i in tqdm(range(int(math.floor(len(validation_data) / batch_size))),
					  mininterval=0.1, desc='  - (Validation)   ', leave=False):
			# prepare data
			batch_edge = validation_data[i * batch_size:(i + 1) * batch_size]
			batch_edge_weight = validation_weight[i * batch_size:(i + 1) * batch_size]

			# if len(y) == 0:
			# 	batch_x, batch_y, batch_w, batch_s = generate_negative(
			# 		batch_x, "test_dict", weight=batch_w, neg_num=neg_num)
			# else:
			# 	batch_y = y[i * batch_size:(i + 1) * batch_size]
			#
			# batch_x, batch_y, batch_w, batch_s = sync_shuffle(
			# 	[batch_x, batch_y, batch_w, batch_s])
			# pred_batch, recon_loss = model(batch_x, return_recon=True)
			# loss = loss_func(pred_batch, batch_y)
			pred_batch, batch_y, loss, recon_loss, batch_w, batch_s = forward_func(
				model, loss_func, batch_edge, batch_edge_weight)

			size_list.append(batch_s)
			pred.append(pred_batch)
			label.append(batch_y)
			weight_list.append(batch_edge_weight)


			recon_total_loss += recon_loss.item()
			bce_total_loss += loss.item()

		pred = torch.cat(pred, dim=0)
		label = torch.cat(label, dim=0)
		size_list = torch.cat(size_list, dim=0)
		# weight_list = torch.cat(weight_list, dim=0)

		acc = accuracy(pred, label, size_list, max_size)
		auc1, auc2 = roc_auc_cuda(label, pred, size_list, max_size)

	return bce_total_loss / (i + 1), recon_total_loss / \
		   (i + 1), acc, auc1, auc2


def train(model,
		  loss,
		  training_data,
		  validation_data,
		  optimizer,
		  epochs,
		  batch_size):
	valid_accus = [0]
	edges, edge_weight = training_data
	training_data_generator = DataGenerator(
		edges, edge_weight, int(batch_size),1000, min_size=min_size, max_size=max_size)
	start = time.time()
	for epoch_i in range(epochs):

		save_embeddings(model, True)
		print('[ Epoch', epoch_i, 'of', epochs, ']')

		start = time.time()
		edges_part, edge_weight_part = training_data_generator.next_iter()
		training_data_new = edges_part, edge_weight_part

		bce_loss, recon_loss, train_accu, auc1, auc2 = train_epoch(model, loss, training_data_new, optimizer,
																   batch_size)

		print(
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
		valid_aupr_final = float(valid_auc2.split(" ")[-2])
		valid_accus += [valid_aupr_final]

		checkpoint = {
			'model_link': model.state_dict(),
			'epoch': epoch_i}

		if valid_aupr_final >= max(valid_accus):
			torch.save(checkpoint, os.path.join(temp_dir, model_name))
			torch.save(model, os.path.join(temp_dir, "model2load"))

		torch.cuda.empty_cache()

	checkpoint = torch.load(os.path.join(temp_dir, model_name))
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
	# flag = False
	# for i in range(len(temp)):
	# 	for j in [-1, 0, 1]:
	# 		a = np.copy(temp)
	# 		a[i] += j
	# 		a.sort()
	# 		if tuple(a) in dict:
	# 			flag = True
	# 			break
	# 	if flag:
	# 		break
	# return flag


def generate_negative(x, dict1, weight="", neg_num=1):
	if len(weight) == 0:
		weight = torch.ones(len(x), dtype=torch.float)
	if dict1 == 'train_dict':
		dict1 = train_dict
	elif dict1 == 'test_dict':
		dict1 = test_dict

	change_num_list = [[] for i in range(max_size + 1)]
	for s in range(min_size, max_size + 1):
		change_num = np.random.binomial(s, 0.5, int(len(x) * (math.ceil(neg_num) * 2)))
		change_num = change_num[change_num != 0]

		change_num_list[s] = list(change_num)

	neg_list = []
	new_x = []
	new_index = []
	neg_weight = []
	size_list = []
	size_neg_list = []

	for j, sample in enumerate(x):
		for i in range(int(math.ceil(neg_num))):

			decompose_sample = np.copy(sample)
			list1 = change_num_list[decompose_sample.shape[-1]]
			change_num = list1.pop()
			changes = np.random.choice(np.arange(decompose_sample.shape[-1]), change_num, replace=False)
			temp = np.copy(decompose_sample)
			trial = 0
			while neighbor_check(temp, dict1[(len(temp))]):
				temp = np.copy(decompose_sample)
				# trial += 1
				# if trial >= 10000:
				# 	temp = ""
				# 	break

				for change in changes:
					if temp[change] not in node2chrom:
						print(temp, decompose_sample)
					chrom = node2chrom[temp[change]]
					start, end = chrom_range[chrom]

					temp[change] = int(
						math.floor(
							(end - start) * random.random())) + start


				temp = list(set(temp))

				if len(temp) < len(decompose_sample):
					temp = np.copy(decompose_sample)
					continue

				temp.sort()
				dis_list = []
				for k in range(len(temp) - 1):
					dis_list.append(temp[k + 1] - temp[k])
				if np.min(dis_list) <= min_dis:
					temp = np.copy(decompose_sample)
			
			if i == 0:
				size_list.append(len(decompose_sample))
			if len(temp) > 0:
				neg_list.append(temp)
				size_neg_list.append(len(temp))
				neg_weight.append(weight[j])
				
	pos_weight = weight
	pos_weight = torch.tensor(pos_weight).to(device)
	size_list = torch.tensor(size_list + size_neg_list)
	pos_part = np2tensor_hyper(list(x), dtype=torch.long)
	neg = np2tensor_hyper(neg_list, dtype=torch.long)
	if type(pos_part) == list:
		pos_part = pad_sequence(pos_part, batch_first=True, padding_value=0)
		neg = pad_sequence(neg, batch_first=True, padding_value=0)

	if len(neg) == 0:
		neg = torch.zeros((1, pos_part.shape[-1]),dtype=torch.long, device=device)
	pos_part = pos_part.to(device)
	neg = neg.to(device)
	if task_mode == 'class':
		y = torch.cat([torch.ones((len(pos_part), 1), device=device),
					   torch.zeros((len(neg), 1), device=device)], dim=0)
		w = torch.cat([torch.ones((len(pos_part), 1), device=device) * pos_weight.view(-1, 1),
					   torch.ones((len(neg), 1), device=device)])
		x = torch.cat([pos_part, neg])
	elif task_mode == 'regress':
		w = torch.cat([torch.ones((len(pos_part), 1), device=device),
					   torch.ones((len(neg), 1), device=device)], dim=0)
		y = torch.cat([torch.ones((len(pos_part), 1), device=device) * pos_weight.view(-1, 1),
					   torch.zeros((len(neg), 1), device=device)])
		x = torch.cat([pos_part, neg])
	else:
		print ("Wrong task mode")
		raise EOFError

	return x, y, w, size_list


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
			embed = embed.detach().cpu().numpy()
			embeddings.append(embed)

		embeddings = np.concatenate(embeddings, axis=0)[:, 0, :]
		np.save("../embeddings.npy" , embeddings)

	torch.cuda.empty_cache()
	return embeddings


def predict(model, input):
	model.eval()
	output = []
	new_batch_size = int(1e5)
	with torch.no_grad():
		for j in trange(math.ceil(len(input) / new_batch_size)):
			x = input[j * new_batch_size:min((j + 1) * new_batch_size, len(input))]
			x = np2tensor_hyper(x, dtype=torch.long)
			x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
			output.append(model(x).detach().cpu().numpy())
	output = np.concatenate(output, axis=0)
	torch.cuda.empty_cache()
	return output


def get_attributes():
	attribute_all = []
	for i in range(len(num)):
		chrom = np.zeros((num[i], len(chrom_list)))
		chrom[:, i] = 1
		coor = np.arange(num[i]).reshape((-1, 1)).astype('float32')
		coor /= num[0]
		attribute = np.concatenate([chrom, coor], axis=-1)
		attribute_all.append(attribute)
	
	attribute_all = np.concatenate(attribute_all, axis=0)
	attribute_dict = np.concatenate([np.zeros((1, attribute_all.shape[-1])), attribute_all], axis=0).astype(
		'float32')
	
	print("attribute_dict", attribute_dict.shape)
	return attribute_dict


	
config = get_config()
bottle_neck = config['embed_dim']
size_list = config['k-mer_size']
min_size, max_size = int(np.min(size_list)), int(np.max(size_list))
temp_dir = config['temp_dir']

quantile_cutoff_for_positive = config['quantile_cutoff_for_positive']
quantile_cutoff_for_unlabel = config['quantile_cutoff_for_unlabel']

min_dis = config['min_distance']
chrom_list = config['chrom_list']
neg_num = 3
batch_size = 96
loss = F.binary_cross_entropy_with_logits
model_name = 'model.chkpt'
current_time = datetime.datetime.now()
task_mode = 'class'



neighbor_mask = []

chrom_range = np.load(os.path.join(temp_dir,"chrom_range.npy"))
node2chrom = np.load(os.path.join(temp_dir,"node2chrom.npy"), allow_pickle=True).item()
num = []
for v in chrom_range:
	num.append(v[1] - v[0])

num_list = np.cumsum(num)
zero_num_list = np.array([0] + list(num_list))
print("Node type num", num)

data_list = []
weight_list = []
from sklearn.preprocessing import QuantileTransformer
for size in size_list:
	data = np.load(os.path.join(temp_dir,"all_%d_counter.npy" % size)).astype('int')
	weight = np.load(os.path.join(temp_dir,"all_%d_freq_counter.npy" % size)).astype('float32')
	print("before filter", "size", size, "length", len(data))
	weight = QuantileTransformer(n_quantiles=1000, output_distribution='uniform').fit_transform(weight.reshape((-1,1))).reshape((-1))
	mask = weight > quantile_cutoff_for_positive
	# mask = weight >= cutoff
	data = data[mask]
	weight = weight[mask]
	print("after filter", "size", size, "length", len(data))
	for datum in data:
		data_list.append(datum)
	weight_list.append(weight)

data = np.array(data_list)
weight = np.concatenate(weight_list,axis = 0)


embeddings_initial = []
inter_initial = np.load(os.path.join(temp_dir, "inter_adj.npy")).astype('float32')
adj = np.load(os.path.join(temp_dir, "intra_adj.npy")).astype('float32')
for v in chrom_range:
	temp = adj[v[0] - 1:v[1] - 1, v[0] - 1:v[1] - 1]
	temp = np.corrcoef(temp).astype('float32')
	temp[np.isnan(temp)] = 0.0
	print (temp.shape)
	embeddings_initial.append(temp)


attribute_dict = get_attributes()

num = torch.as_tensor(num)
num_list = torch.as_tensor(num_list)
print(num, num_list)

compress = True
# Note that, no matter how many node types are here, make sure the
# hyperedge (N1,N2,N3,...) has id, N1 < N2 < N3...
train_dict = test_dict = [set() for i in range(max_size+1)]

index = np.arange(len(data))

print ("weight",weight)
weight /= np.mean(weight)
weight *= neg_num
print (weight)

np.random.shuffle(index)
split = int(0.8 * len(index))
train_data = data[index[:split]]
test_data = data[index[split:]]
train_weight = weight[index[:split]]
test_weight = weight[index[split:]]

print("train data amount", len(train_data))

print("dict_size", len(train_dict[-1]), len(test_dict[-1]))

node_embedding = MultipleEmbedding(
	embeddings_initial,
	bottle_neck,
	False,
	num_list, chrom_range, inter_initial).to(device)

classifier_model = Classifier(
	n_head=8,
	d_model=bottle_neck,
	d_k=bottle_neck,
	d_v=bottle_neck,
	node_embedding=node_embedding,
	diag_mask=True,
	bottle_neck=bottle_neck,
	attribute_dict=attribute_dict).to(device)

save_embeddings(classifier_model, True)


params_list = list(classifier_model.parameters())

optimizer = torch.optim.AdamW(params_list, lr=1e-3, amsgrad=False)

model_parameters = filter(lambda p: p.requires_grad, params_list)
params = sum([np.prod(p.size()) for p in model_parameters])
print("params to be trained", params)


alpha = 0.0
beta = 1.0
train(classifier_model,
	  loss=loss,
	  training_data=(train_data, train_weight),
	  validation_data=(test_data, test_weight),
	  optimizer=[optimizer], epochs=3, batch_size=batch_size)


data_list = []
weight_list = []
from sklearn.preprocessing import QuantileTransformer
for size in size_list:
	data = np.load(os.path.join(temp_dir,"all_%d_counter.npy" % size)).astype('int')
	weight = np.load(os.path.join(temp_dir,"all_%d_freq_counter.npy" % size)).astype('float32')
	print("before filter", "size", size, "length", len(data))
	weight = QuantileTransformer(n_quantiles=1000, output_distribution='uniform').fit_transform(weight.reshape((-1,1))).reshape((-1))
	mask = weight > quantile_cutoff_for_unlabel
	data = data[mask]
	weight = weight[mask]
	print("after filter", "size", size, "length", len(data))
	for datum in data:
		data_list.append(datum)
	weight_list.append(weight)
	
dict_data = np.array(data_list)

test_dict = build_hash(dict_data, compress=compress, max_size=max_size,
								 min_size=min_size)

train_dict = test_dict

print ("Finish building Dict")

optimizer = torch.optim.AdamW(params_list, lr=1e-3, amsgrad=False)
alpha = 1.0
beta = 0.001

train(classifier_model,
	  loss=loss,
	  training_data=(train_data, train_weight),
	  validation_data=(test_data, test_weight),
	  optimizer=[optimizer], epochs=30, batch_size=batch_size)

checkpoint = torch.load(os.path.join(temp_dir, model_name))
classifier_model.load_state_dict(checkpoint['model_link'])

save_embeddings(classifier_model, True)
torch.save(classifier_model, os.path.join(temp_dir, "model2load"))

