import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from itertools import combinations
import time
import os
import argparse

def parse_args():
	# Parses the node2vec arguments.
	parser = argparse.ArgumentParser(description="Hyper-SAGNN")

	parser.add_argument('-n','--num', type=int, default=0)
	parser.add_argument('-s', '--size', type=int, default=3)
	args = parser.parse_args()
	return args

args = parse_args()
size = args.size
thresh_list = [[2,3],[3,5],[5,8],[8,12]]
# thresh_list = [[0,2]]
def build_dict(i):
	hash_dict = {}
	for j,datum in enumerate(tqdm(data)):
		if i not in data_set_list[j]:
			continue
		combs = combinations(datum[datum > i + 5], size - 1)
		for comb in combs:
			new_comb = (i,) + comb
			if new_comb in hash_dict:
				hash_dict[new_comb] += 1
			else:
				hash_dict[new_comb] = 1
	new_hash = hash_dict

	np.save("./dict_%dnode/hash_tuples_%d.npy" % (size,i), new_hash)
	del hash_dict,new_hash
	return 0


chrom_range = np.load("../data/SPRITE/chrom_range.npy")


# data = np.load("../data/SPRITE/edge_list.npy", allow_pickle=True)
# new_data = []
# for datum in data:
# 	if (len(datum) >= size) & (len(datum) < 25):
# 		new_data.append(np.array(datum))
# data = np.array(new_data)
#
# np.save("./shrink_SPRITE.npy",data)
data= np.load("./shrink_SPRITE.npy",allow_pickle=True)
data_set_list = [set(datum) for datum in tqdm(data)]
from concurrent.futures import ProcessPoolExecutor, as_completed


process_list = []
MAX_WORKER = 10
pool = ProcessPoolExecutor(max_workers=MAX_WORKER)

node_list = np.arange(2746)
job_iter = iter(node_list)
jobs_left = len(node_list)
while jobs_left:
	for i in job_iter:
		process_list.append(pool.submit(build_dict,i))
		if len(process_list) > MAX_WORKER * 1.3:
			break
		time.sleep(1)
	start = time.time()
	for p in as_completed(process_list):
		a = p.result()
		process_list.remove(p)
		del p
		jobs_left -= 1
		print (jobs_left)

		if time.time() - start > 5:
			break

pool.shutdown(wait=True)


for thres in thresh_list:
	if not os.path.exists("./%d_freq/%d/" % (size,thres[0])):
		os.mkdir("./%d_freq/%d/" % (size,thres[0]))
if not os.path.exists("./%d_freq/upper/" % size):
	os.mkdir ("./%d_freq/upper/" % size)
def dict2freq(i):
	print ("start %d" % i)
	hash_dict = np.load("./dict_%dnode/hash_tuples_%d.npy" % (size,i), allow_pickle=True).item()
	keys = np.array(list(hash_dict.keys()))
	if len(keys) > 0:
		dis_list = np.zeros((len(keys),size-1))


		for j in range(size - 1):
			dis_list[:,j] = keys[:,j+1] - keys[:,j]

		dis_list = np.min(dis_list,axis = -1)
		old_length  = len(keys)
		keys = keys[dis_list > 5]
		print (old_length,len(keys))
	freq = np.array([hash_dict[tuple(k)] for k in tqdm(keys)])

	for thres in thresh_list:
		temp = keys[(freq >= thres[0]) & (freq < thres[1])]
		np.save("./%d_freq/%d/%d.npy" % (size,thres[0], i), temp)

	temp = keys[freq >= thres[1]]
	np.save("./%d_freq/upper/%d.npy" % (size,i), temp)
	del hash_dict, temp



from concurrent.futures import ProcessPoolExecutor, as_completed


process_list = []
MAX_WORKER = 100
pool = ProcessPoolExecutor(max_workers=MAX_WORKER)
node_list = np.arange(2746)
job_iter = iter(node_list)
jobs_left = len(node_list)
while jobs_left:
	for i in job_iter:
		process_list.append(pool.submit(dict2freq,i))
		if len(process_list) > MAX_WORKER * 1.3:
			break
		time.sleep(1)
	start = time.time()
	for p in as_completed(process_list):
		a = p.result()
		process_list.remove(p)
		del p
		jobs_left -= 1
		time.sleep(1)

		if time.time() - start > 10:
			break

pool.shutdown(wait=True)



for thres in thresh_list:
	list1 = []
	for i in trange(2746):
		temp = np.load("./%d_freq/%d/%d.npy" % (size,thres[0], i))
		if len(temp) > 0:
			list1.append(temp)
	list1 = np.concatenate(list1,axis = 0)
	print (list1.shape)
	np.save("./%d_%d_%d.npy" %(thres[0], thres[1],size),list1)
list1 = []
for i in trange(2746):
	if i == 0:
		continue
	temp = np.load("./%d_freq/upper/%d.npy" % (size,i))
	if len(temp) > 0:
		list1.append(temp)
list1 = np.concatenate(list1,axis = 0)
print (list1.shape)
np.save("./upper_%d.npy" % size,list1)