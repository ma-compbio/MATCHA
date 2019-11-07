import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from concurrent.futures import as_completed, ProcessPoolExecutor
from pybloom_live import ScalableBloomFilter
from copy import copy,deepcopy
from itertools import combinations
from pybloomfilter import BloomFilter
import os
import time
def add_padding_idx(vec):
	if len(vec.shape) == 1:
		return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
						 for v in tqdm(vec)])
	else:
		vec = np.asarray(vec) + 1
		vec = np.sort(vec, axis=-1)
		return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	vec = np.asarray(vec)
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype = dtype)


def walkpath2str(walk):
	return [list(map(str, w)) for w in tqdm(walk)]


def roc_auc_cuda(y_true, y_pred, size_list, max_size):
	roc_str, aupr_str = "", ""
	try:
		for s in np.unique(size_list):
			y_t = (y_true[size_list == s] > 0.5).float().cpu().detach().numpy().reshape((-1, 1))
			y_p = y_pred[size_list == s].cpu().detach().numpy().reshape((-1, 1))
			roc, aupr =  roc_auc_score(
				y_t, y_p), average_precision_score(
				y_t, y_p)
			roc_str += "%d %.3f " % (s, roc)
			aupr_str += "%d %.3f " % (s, aupr)
		return roc_str, aupr_str
	except BaseException:
		return 0.0, 0.0


def accuracy(output, target, size_list = None, max_size = None):
	acc_str = ""
	if size_list is not None:
		for s in np.unique(size_list):
			pred = output[size_list == s] >= 0.5
			truth = target[size_list == s] >= 0.5
			acc = torch.sum(pred.eq(truth))
			acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
			acc_str += "%d %.3f " % (s, acc)
	else:
		pred = output >= 0.5
		truth = target >= 0.5
		acc = torch.sum(pred.eq(truth))
		acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
		acc_str += "%.3f " % (acc)
	return acc_str


def build_hash(data,compress,max_size,min_size,fname):
	# if os.path.isfile("../data/SPRITE/%s_dict_%d" %(fname, 0)):
	# 	print ("existing dict")
	# 	dict_list = [BloomFilter.open("../data/SPRITE/%s_dict_%d" %(fname, i)) for i in range(max_size + 1)]
	# else:
	# 	dict_list = []
	# 	for i in range(max_size + 1):
	# 		if i <= 2:
	# 			dict_list.append(BloomFilter(10, 1e-3, "../data/SPRITE/%s_dict_%d" %(fname, i)))
	# 		else:
	# 			dict_list.append(BloomFilter(3e9, 1e-3, "../data/SPRITE/%s_dict_%d" % (fname, i)))
	
	dict_list = []
	for i in range(max_size + 1):
		if i <= 2:
			dict_list.append(BloomFilter(10, 1e-3))
			# dict_list.append(set())
		else:
			dict_list.append(BloomFilter(5e8, 1e-3))
			# dict_list.append(set())
	print (len(dict_list))
	data_list = [[] for i in range(max_size + 1)]
	for datum in tqdm(data):
		# datum = np.array(datum).astype('int')
		if (min_size < 0) or (len(datum) >= min_size):
			# if len(datum) >= 100:
			# 	continue
			# datum.sort()
			# for j in range(min_size, min(len(datum), max_size) + 1):
				# combs = combinations(datum, j)
				# data_list[j].append(combs)
			data_list[len(datum)].append(tuple(datum))

		if len(data_list[min_size]) > 1e7:
			start = time.time()
			for i in range(max_size + 1):
				dict_list[i].update(data_list[i])
				# for combs in data_list[i]:
				# 	dict_list[i].update(combs)
					# for d in combs:
					# 	if d in dict_list[i]:
					# 		# final_data.append(d)
					# 		continue
					# 	else:
					# 		# final_data.append(d)
					# 		dict_list[i].add(d)
			print(len(dict_list[min_size]) / dict_list[min_size].capacity, "%.2f s" %(time.time() - start))
			data_list = [[] for i in range(max_size + 1)]

	
	for i in range(max_size + 1):
		for combs in data_list[i]:
			dict_list[i].add(combs)
	
	
	print (len(dict_list[-1]))
	length_list = [len(dict_list[i]) for i in range(len(dict_list))]
	print (length_list)
	np.save("../data/SPRITE/length.npy", length_list)
	return dict_list


def parallel_build_hash(data, func, args, num, initial=None, compress=False, max_size=-1):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	data = np.array_split(data, cpu_num * 1)
	dict1 = deepcopy(initial)
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	if func == 'build_hash2':
		func = build_hash2
	if func == 'build_hash3':
		func = build_hash3

	for datum in data:
		process_list.append(pool.submit(func, datum, compress, max_size))

	for p in as_completed(process_list):
		a = p.result()
		if compress:
			dict1 = dict1.union(a)
		else:
			dict1.update(a)
		del a
	pool.shutdown(wait=True)

	# if args.data in ['schic','ramani']:
	# 	print (num[0])
	# 	new_list_of_set = [set() for i in range(int(num[0]+1))]
	# 	for s in dict1:
	# 		try:
	# 			new_list_of_set[s[0]].add(s)
	# 		except:
	# 			print (s)
	# 			raise EOFError
	# 	dict1 = new_list_of_set
	return dict1

def sync_shuffle(sample_list, max_num = -1):
	index = torch.randperm(len(sample_list[0]))
	if max_num > 0:
		index = index[:max_num]
	new_list = []
	for s in sample_list:
		new_list.append(s[index])
	return new_list


def pass_(x):
    return x


def generate_outlier_part(data, dict_pair, k=20):
	inputs = []
	negs = []
	
	for e in tqdm(data):
		point = int(np.where(e == 0)[0])
		start = 0 if point == 0 else int(num_list[point - 1])
		end = int(num_list[point])
		
		count = 0
		trial = 0
		while count < k:
			trial += 1
			if trial >= 100:
				break
			j = np.random.randint(start, end) + 1
			condition = [(j, n) in dict_pair for n in e]
			if np.sum(condition) > 0:
				continue
			else:
				temp = np.copy(e)
				temp[point] = j
				inputs.append(temp)
				negs.append(point)
				count += 1
	inputs, index = np.unique(inputs, axis=0, return_index=True)
	negs = np.array(negs)[index]
	return np.array(inputs), np.array(negs)


def check_outlier(model, data_):
	data, negs = data_
	bs = 1024
	num_of_batches = int(np.floor(data.shape[0] / bs)) + 1
	k = 3
	outlier_prec = torch.zeros(k).to(device)
	
	model.eval()
	with torch.no_grad():
		for i in tqdm(range(num_of_batches)):
			inputs = data[i * bs:(i + 1) * bs]
			neg = negs[i * bs:(i + 1) * bs]
			outlier = model(inputs, get_outlier=k)
			outlier_prec += (outlier.transpose(1, 0) == neg).sum(dim=1).float()
		# for kk in range(k):
		# 	outlier_prec[kk] += (outlier[:,kk].view(-1)==neg).sum().float()
		outlier_prec = outlier_prec.cumsum(dim=0)
		outlier_prec /= data.shape[0]
		for kk in range(k):
			print("outlier top %d hitting: %.5f" % (kk + 1, outlier_prec[kk]))


class Word2Vec_Skipgram_Data_Empty(object):
	"""Word2Vec model (Skipgram)."""
	
	def __init__(self):
		return
	
	def next_batch(self):
		"""Train the model."""
		
		return 0, 0, 0, 0, 0
	