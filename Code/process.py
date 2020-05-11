import numpy as np
import pandas as pd
import math
from tqdm import tqdm, trange
import sys, os
from utils import *
import h5py


def build_node_dict():
	tab = pd.read_table(chrom_size, header=None, sep="\t")
	tab.columns = ['chr', 'size']
	print(tab)

	bin2node = {}
	node2bin = {}
	node2chrom = {}
	chrom_range = []
	count = 1

	for j, chrom in enumerate(chrom_list):
		size = np.max(tab['size'][tab['chr'] == chrom])
		max_bin_chrom = math.ceil(size / res)

		temp = [count]
		for i in range(max_bin_chrom + 1):
			bin_ = "%s:%d" % (chrom, i * res)
			bin2node[bin_] = count
			node2bin[count] = bin_
			node2chrom[count] = j
			count += 1
		temp.append(count)
		chrom_range.append(temp)
	print(chrom_range)
	
	np.save(os.path.join(temp_dir, "chrom_range.npy"), chrom_range)
	np.save(os.path.join(temp_dir, "bin2node.npy"), bin2node)
	np.save(os.path.join(temp_dir, "node2chrom.npy"), node2chrom)
	np.save(os.path.join(temp_dir,"node2bin.npy"), node2bin)
	
	
def parse_file():
	file1 = open(cluster_path, "r")
	
	bin2node = np.load(os.path.join(temp_dir, "bin2node.npy"), allow_pickle=True).item()
	
	line = file1.readline()
	count = 0
	final = []
	
	while line:
		info_list = line.strip().split("\t")[1:]
		temp = []
		if (len(info_list) < 2) or (len(info_list) > max_cluster_size * 50):
			line = file1.readline()
			continue
		
		for info in info_list:
			try:
				chrom, bin_ = info.split(":")
			except:
				print (info)
				raise EOFError
			if chrom not in chrom_list:
				continue
			bin_ = int(math.floor(int(bin_) / res)) * res
			bin_ = "%s:%d" % (chrom, bin_)
			node = bin2node[bin_]
			temp.append(node)
		temp = list(set(temp))
		
		
		if len(temp) > max_cluster_size:
			line = file1.readline()
			continue
		
		temp.sort()
		count += 1
		if count % 100 == 0:
			print("%d\r" % count, end="")
			sys.stdout.flush()
		if len(temp) > 1:
			final.append(temp)
		
		line = file1.readline()

	np.save(os.path.join(temp_dir, "edge_list.npy"), final)
	

def edgelist2adj():
	edge_list = np.load(os.path.join(temp_dir, "edge_list.npy"), allow_pickle=True)
	chrom_range = np.load(os.path.join(temp_dir, "chrom_range.npy"), allow_pickle=True)
	
	node_num = int(np.max(chrom_range))
	print(node_num)
	
	adj = np.zeros((node_num - 1, node_num - 1))
	
	for e in tqdm(edge_list):
		for i in e:
			for j in e:
				if i!=j:
					adj[i-1,j-1] += 1
	print (adj)
	np.save(os.path.join(temp_dir, "edge_list_adj.npy"), adj)

def parse_cool_contact():
	f = h5py.File(mcool_path, "r")
	f = f['resolutions']
	f = f[str(res)]
	
	bin2node = np.load(os.path.join(temp_dir, "bin2node.npy"), allow_pickle=True).item()
	node2chrom = np.load(os.path.join(temp_dir, "node2chrom.npy"), allow_pickle=True).item()
	
	cool_bin_info_chrom = np.array(f['bins']['chrom'])
	cool_bin_info_start = np.array(f['bins']['start'])
	chrom_name = np.array(f['chroms']['name']).astype('str')
	print ("chrom_name", chrom_name)
	
	cool_index2node = {}
	print ("Building dict to map cool bin to MATCHA node id")
	for i in trange(len(cool_bin_info_chrom)):
		chrom = cool_bin_info_chrom[i]
		start =  cool_bin_info_start[i]
		chrom = chrom_name[chrom]
		
		if chrom not in chrom_list:
			print (chrom)
			continue
		bin = "%s:%d" %(chrom, start)
		
		node = bin2node[bin]
		cool_index2node[i] = node
		
	chrom_range = np.load(os.path.join(temp_dir, "chrom_range.npy"))

	node_num = int(np.max(chrom_range))
	print(node_num)

	intra_adj = np.zeros((node_num - 1, node_num - 1))
	inter_adj = np.zeros((node_num - 1, node_num - 1))
	
	
	cool_index_bin1 = np.array(f['pixels']['bin1_id'])
	cool_index_bin2 = np.array(f['pixels']['bin2_id'])
	if 'balanced' in f['pixels'].keys():
		cool_count = np.array(f['pixels']['balanced'])
	else:
		cool_count = np.array(f['pixels']['count'])
	
	print ("Building adjacency matrxi from mcool file")
	for i in trange(len(cool_index_bin1)):
		index1 = cool_index_bin1[i]
		index2 = cool_index_bin2[i]
		if (not index1 in cool_index2node) or (not index2 in cool_index2node):
			continue
		# minus 1 because, our node id starts at 1
		node1 = cool_index2node[index1] - 1
		node2 = cool_index2node[index2] - 1
		count = float(cool_count[i])
		
		if not np.isnan(count):
			chrom1 = node2chrom[node1 + 1]
			chrom2 = node2chrom[node2 + 1]
			
			
			if chrom1 == chrom2:
				intra_adj[node1, node2] += count
				intra_adj[node2, node1] += count
			else:
				inter_adj[node1, node2] += count
				inter_adj[node2, node1] += count

	print(intra_adj, inter_adj)
	np.save(os.path.join(temp_dir, "intra_adj.npy"), intra_adj)
	np.save(os.path.join(temp_dir, "inter_adj.npy"), inter_adj)

def build_subcompartment_label():
	tab = pd.read_table("../gm12878_subcompartments_hg38.bed", sep="\t", header=None)
	tab = tab[tab.columns[:4]]
	bin2node = np.load(os.path.join(temp_dir, "bin2node.npy"), allow_pickle=True).item()
	tab.columns = ['chrom', 'start', 'end', 'label']
	print(tab)
	chrom_range = np.load(os.path.join(temp_dir, "chrom_range.npy"), allow_pickle=True)
	
	node_num = int(np.max(chrom_range))
	print(node_num)
	
	state_dict = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'B3': 4}
	
	label_list = np.ones((node_num, 10)) * -1
	for i in range(len(tab)):
		chrom = tab['chrom'][i]
		start = tab['start'][i]
		end = tab['end'][i]
		label = tab['label'][i]
		if label in state_dict:
			label = state_dict[label]
		else:
			label = -1
		
		start = int(math.floor(start / 100000))
		end = int(math.floor(end / 100000))
		
		for j in range(start, end + 1):
			larger_bin = int(math.floor(j / 10))
			coord = "%s:%d" % (chrom, larger_bin * 1000000)
			if coord in bin2node:
				coord = bin2node[coord]
				label_list[coord, j % 10] = label
	
	print(label_list, np.min(label_list), np.max(label_list))
	
	final = []
	
	for vec in tqdm(label_list):
		unique, count = np.unique(vec, return_counts=True)
		if np.max(count) >= 6:
			pick = unique[np.argmax(count)]
			final.append(pick)
		else:
			final.append(-1)
	final = np.array(final)
	print(final, np.sum(final != -1))
	final = final[1:]
	np.save("../subcompartment_label_hg38_1Mb.npy", final)
	
	
config = get_config()
res = config['resolution']
chrom_list = config['chrom_list']
temp_dir = config['temp_dir']
cluster_path = config['cluster_path']
mcool_path = config['mcool_path']
chrom_size = config['chrom_size']
max_cluster_size = config['max_cluster_size']
if not os.path.exists(temp_dir):
	os.mkdir(temp_dir)
build_node_dict()
parse_file()
# edgelist2adj()
parse_cool_contact()

# build_subcompartment_label()