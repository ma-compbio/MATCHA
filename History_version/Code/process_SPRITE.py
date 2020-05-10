import numpy as np
import pandas as pd
import math
from tqdm import tqdm, trange
res = 1000000
chrom_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
				"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
				"chr18", "chr19", "chr20","chr21","chr22"]
def filter():
	file1 = open("../../SPRITE/4DNFIBEVVTN5.clusters","r")
	result = open("../../SPRITE/filtered.txt","w")


	line = file1.readline()

	while line:
		info = line.strip().split("\t")
		if len(info) <= 2:
			line = file1.readline()
			continue
		else:
			result.write("\t".join(info[1:])+"\n")

		line = file1.readline()

	result.close()


def build_node_dict():
	tab = pd.read_table("../../SPRITE/hg38.chrom.sizes.txt",header = None,sep = "\t")
	tab.columns = ['chr','size']
	print (tab)
	
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
			bin_ = "%s:%d" %(chrom, i * res)
			bin2node[bin_] = count
			node2bin[count] = bin_
			node2chrom[count] = j
			count += 1
		temp.append(count)
		chrom_range.append(temp)
	print (chrom_range)
	np.save("../data/SPRITE/chrom_range.npy",chrom_range)
	np.save("../data/SPRITE/bin2node.npy", bin2node)
	np.save("../data/SPRITE/node2chrom.npy",node2chrom)
	np.save("../data/SPRITE/node2bin.npy",node2bin)

def parse_file():
	result = open("../../SPRITE/filtered.txt", "r")
	line = result.readline()
	bin2node = np.load("../data/SPRITE/bin2node.npy",allow_pickle=True).item()
	node2bin = np.load("../data/SPRITE/node2bin.npy",allow_pickle=True).item()
	node2chrom = np.load("../data/SPRITE/node2chrom.npy",allow_pickle=True).item()
	final = []
	while line:
		info_list = line.strip().split("\t")
		temp = []
		if len(info_list) > 1000:
			line = result.readline()
			continue
		for info in info_list:
			chrom, bin_ = info.split(":")
			if chrom not in chrom_list:
				continue
			bin_ = int(math.floor(int(bin_) / res)) * res
			bin_ = "%s:%d" %(chrom,bin_)
			node = bin2node[bin_]
			temp.append(node)
		temp = list(set(temp))
		temp.sort()

		if len(temp) > 1:
			final.append(temp)


		line = result.readline()

	np.save("../data/SPRITE/edge_list.npy", final)

	final = np.load("../data/SPRITE/edge_list.npy", allow_pickle=True)
	chrom_range = np.load("../data/SPRITE/chrom_range.npy", allow_pickle=True)
	node_freq = np.zeros((np.max(chrom_range)))
	for e in tqdm(final):
		if len(e) > 25:
			continue

		for n in e:
			node_freq[n] += 1
	print (node_freq)

	drop_list = np.where(node_freq <= 50)[0]
	print (drop_list, len(drop_list))

	node2newnode = {}
	dropnode2newnode = {}
	newnode2chrom = {}

	count = 1
	for n in range(np.max(chrom_range)):
		if n == 0:
			continue
		elif n in drop_list:
			dropnode2newnode[n] = count
		else:
			node2newnode[n] = count
			count += 1
	dropnode2newnode[n+1] = count
	print ("remap")

	new_node2bin = {}
	new_bin2node = {}

	for node in node2bin:
		if node in node2newnode:
			new_node2bin[node2newnode[node]] = node2bin[node]
			new_bin2node[node2bin[node]] = node2newnode[node]
			newnode2chrom[node2newnode[node]] = node2chrom[node]

	np.save("../data/SPRITE/bin2node.npy", new_bin2node)
	np.save("../data/SPRITE/node2bin.npy", new_node2bin)
	np.save("../data/SPRITE/node2chrom.npy", newnode2chrom)

	new_final = []
	for e in tqdm(final):
		temp = []
		for n in e:
			if n in node2newnode:
				temp.append(node2newnode[n])
		if len(temp) >= 2:
			new_final.append(temp)
	final = new_final
	new_chrom_range = []
	for v in chrom_range:
		temp = []
		if v[0] in node2newnode:
			temp.append(node2newnode[v[0]])
		else:
			temp.append(dropnode2newnode[v[0]])

		if v[1] in node2newnode:
			temp.append(node2newnode[v[1]])
		else:
			temp.append(dropnode2newnode[v[1]])

		new_chrom_range.append(temp)
	print (chrom_range,new_chrom_range)

	# print (final)
	np.save("../data/SPRITE/edge_list.npy",final)
	np.save("../data/SPRITE/chrom_range.npy", new_chrom_range)

		
def parse_cool_contact():
	file = pd.read_table("../data/SPRITE/SPRITE_contact.txt",sep = "\t")
	bin2node = np.load("../data/SPRITE/bin2node.npy", allow_pickle=True).item()
	chrom_range = np.load("../data/SPRITE/chrom_range.npy")

	node_num = int(np.max(chrom_range))
	print (node_num)

	intra_adj = np.zeros((node_num - 1,node_num - 1))
	inter_adj = np.zeros((node_num - 1,node_num - 1))
	for i in trange(len(file)):
		chrom1 = file['chrom1'][i]
		start1 = file['start1'][i]
		chrom2 = file['chrom2'][i]
		start2 = file['start2'][i]
		
		if chrom1 not in chrom_list or chrom2 not in chrom_list:
			continue
		
		w = file['balanced'][i]

		if not np.isnan(w):
			bin1 = "%s:%d" %(chrom1,start1)
			bin2 = "%s:%d" %(chrom2,start2)
			if bin1 in bin2node and bin2 in bin2node:
				node1 = bin2node[bin1] - 1
				node2 = bin2node[bin2] - 1
				if chrom1 == chrom2:
					intra_adj[node1, node2] += w
					intra_adj[node2,node1] += w
				else:
					inter_adj[node1, node2] += w
					inter_adj[node2, node1] += w
			else:
				print (bin1,bin2)

	print(intra_adj, inter_adj)
	np.save("../data/SPRITE/intra_adj_SPRITE.npy", intra_adj)
	np.save("../data/SPRITE/inter_adj_SPRITE.npy", inter_adj)


build_node_dict()
parse_file()
parse_cool_contact()