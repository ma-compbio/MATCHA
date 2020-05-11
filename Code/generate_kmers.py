from itertools import combinations
import time
import os
from collections import Counter
import multiprocessing
from utils import *

def build_dict(size, i_list):
	list1 = []
	list1_freq = []
	
	for i in tqdm(i_list):
		hash_counter = Counter()
		for j,index in enumerate(node2usefulindex[i]):
			datum = new_data[index]
			n = len(datum)
			# weight = weights[n]
			combs = combinations(datum[datum > i + min_dis], size - 1)
			keys = np.array(list(combs))

			if len(keys) <= 0:
				continue
				
			if size > 2:
				dis_list = np.zeros((len(keys), size - 2))
	
				for j in range(size - 2):
					dis_list[:, j] = keys[:, j + 1] - keys[:, j]
	
				dis_list_min = np.min(dis_list, axis=-1)
	
				length = len(keys)
				keys = keys[dis_list_min > min_dis]
				
			for comb in keys:
				comb = tuple(comb)
				hash_counter[comb] += 1
				# hash_dict[comb] += weight
		# print (hash_counter)
		
		new_hash = {el:hash_counter[el] for el in hash_counter if hash_counter[el] >= min_freq_cutoff}
		hash_dict = new_hash
		
		keys = list(hash_dict.keys())
		freq = np.array([hash_dict[k] for k in keys])
		keys = np.array(keys)
		if len(keys) > 0:
			keys = np.concatenate([np.ones((len(keys), 1), dtype='int') * i, keys], axis=-1)
		
		temp = keys
		temp_freq = freq
		if len(temp) == 0:
			temp = np.array([])
			temp_freq = np.array([])
		if len(temp) > 0:
			list1.append(temp)
			list1_freq.append(temp_freq)
		
		
		
	if len(list1) > 0:
		list1 = np.concatenate(list1, axis=0)
		list1_freq = np.concatenate(list1_freq, axis=0)
	else:
		list1 = np.array([])
		list1_freq = np.array([])
		
	del new_hash, hash_counter
	return i_list, list1, list1_freq



config = get_config()
max_size = config['max_cluster_size']
k_list = config['k-mer_size']
temp_dir = config['temp_dir']
min_dis = config['min_distance']
min_freq_cutoff = config['min_freq_cutoff']

chrom_range = np.load(os.path.join(temp_dir, "chrom_range.npy"))
node_num = np.max(chrom_range) + 1
print(chrom_range)
data = np.load(os.path.join(temp_dir, "edge_list.npy"), allow_pickle=True)
MAX_WORKER = multiprocessing.cpu_count()

for k in k_list:
	size = k
	
	new_data = []
	for datum in tqdm(data):
		if (len(datum) >= size) & (len(datum) <= max_size):
			new_data.append(np.array(datum))
	new_data = np.array(new_data)
	node2usefulindex = [[] for i in range(node_num)]
	for i, datum in enumerate(tqdm(new_data)):
		for n in datum:
			node2usefulindex[n].append(i)
	node2usefulindex = np.array(node2usefulindex)
	
	process_list = []
	list1 = []
	list1_freq = []
	pool = ProcessPoolExecutor(max_workers=MAX_WORKER)
	
	node_list = np.arange(node_num).astype('int')
	batch_size = 50
	job_iter = np.array_split(node_list, int(len(node_list) / batch_size))
	job_iter = iter(job_iter)
	jobs_left = len(node_list)
	
	while jobs_left > 0:
		for i in job_iter:
			process_list.append(pool.submit(build_dict, size, i))
			time.sleep(0.2)
			if len(process_list) > 1.3*MAX_WORKER:
				break
	
		start = time.time()
		for p in as_completed(process_list):
			a = p.result()
			i_list, temp, temp_freq = a
			jobs_left -= len(i_list)
			list1.append(temp)
			list1_freq.append(temp_freq)
			
			process_list.remove(p)
			del p
			if time.time() - start >= 10:
				break
			print ("jobs_left", jobs_left)
	pool.shutdown(wait=True)
	
	
	if len(list1) > 0:
		list1 = np.concatenate(list1,axis = 0)
		list1_freq = np.concatenate(list1_freq, axis=0)
		print()
		print (list1.shape)
		np.save(os.path.join(temp_dir,"all_%d_counter.npy" % size) ,list1)
		np.save(os.path.join(temp_dir,"all_%d_freq_counter.npy" % size) , list1_freq)
	print ("Quick summarize")
	print ("total data", len(list1_freq))
	for c in [2,3,4,5,6,7,8]:
		print (">= %d" %c, np.sum(list1_freq >= c))