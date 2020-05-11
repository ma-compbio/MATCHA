import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from concurrent.futures import as_completed, ProcessPoolExecutor
from copy import copy, deepcopy
from pybloom_live import BloomFilter
import math
from tqdm import tqdm, trange
import os

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
		return torch.as_tensor(vec, dtype=dtype)


def roc_auc_cuda(y_true, y_pred, size_list, max_size):
	roc_str, aupr_str = "", ""
	try:
		y_t = (y_true > 0.5).float().cpu().detach().numpy().reshape((-1, 1))
		y_p = y_pred.cpu().detach().numpy().reshape((-1, 1))
		roc, aupr = roc_auc_score(
			y_t, y_p), average_precision_score(
			y_t, y_p)
		roc_str += "%s %.3f " % ('all', roc)
		aupr_str += "%s %.3f " % ('all', aupr)
		
		for s in np.unique(size_list):
			y_t = (y_true[size_list == s] > 0.5).float().cpu().detach().numpy().reshape((-1, 1))
			y_p = y_pred[size_list == s].cpu().detach().numpy().reshape((-1, 1))
			roc, aupr = roc_auc_score(
				y_t, y_p), average_precision_score(
				y_t, y_p)
			roc_str += "%s %.3f " % (str(s), roc)
			aupr_str += "%s %.3f " % (str(s), aupr)
		
		return roc_str[:-1], aupr_str[:-1]
	except BaseException:
		return 0.0, 0.0


def accuracy(output, target, size_list=None, max_size=None):
	acc_str = ""
	if size_list is not None:
		for s in np.unique(size_list):
			pred = output[size_list == s] >= 0.5
			truth = target[size_list == s] >= 0.5
			acc = torch.sum(pred.eq(truth))
			acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
			acc_str += "%s %.3f " % (str(s), acc)
	else:
		pred = output >= 0.5
		truth = target >= 0.5
		acc = torch.sum(pred.eq(truth))
		acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
		acc_str += "%.3f " % (acc)
	return acc_str


def build_hash(data, compress, min_size, max_size, capacity=None):
	if capacity is None:
		capacity = len(data) * 5
		capacity = int(math.ceil(capacity)) + 1000
		print("total_capacity", capacity)
	dict_list = []
	for i in range(max_size + 1):
		if i < min_size:
			dict_list.append(BloomFilter(10, 1e-3))
		else:
			dict_list.append(BloomFilter(capacity, 1e-3))
	
	print(len(dict_list))
	for datum in tqdm(data):
		dict_list[len(datum)].add(tuple(datum))
		
	print(len(dict_list[min_size]) / dict_list[min_size].capacity)
	
	print(len(dict_list[-1]))
	length_list = [len(dict_list[i]) for i in range(len(dict_list))]
	print(length_list)
	# np.save("../data/SPRITE/length.npy", length_list)
	return dict_list


def parallel_build_hash(data, func, initial=None, compress=False, min_size=-1, max_size=-1):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	np.random.shuffle(data)
	data = np.array_split(data, min(cpu_num * 1, 32))
	length = len(data)
	dict1 = deepcopy(initial)
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []
	
	if func == 'build_hash':
		func = build_hash
		
	for datum in data:
		process_list.append(pool.submit(func, datum, compress, min_size, max_size, length))
	
	for p in as_completed(process_list):
		a = p.result()
		if dict1 is None:
			dict1 = a
		elif compress:
			for i, d in enumerate(dict1):
				dict1[i] = d.union(a[i])
		else:
			for i, d in enumerate(dict1):
				dict1[i] = d.update(a[i])
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


def sync_shuffle(sample_list, max_num=-1):
	index = torch.randperm(len(sample_list[0]))
	if max_num > 0:
		index = index[:max_num]
	new_list = []
	for s in sample_list:
		new_list.append(s[index])
	return new_list


def pass_(x):
	return x



def get_config():
	c = open("./config.JSON","r")
	return json.load(c)