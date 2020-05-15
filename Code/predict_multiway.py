import numpy as np
from utils import *

import numpy as np
from utils import *
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import QuantileTransformer
import os
import sys
import torch.nn.functional as F
import random
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="predict multi-way interactions")
	parser.add_argument("-i", "--file", type=str)
	parser.add_argument("-o", "--output", type=str, default="./output.txt")
	
	return parser.parse_args()


def parse_file(filepath):
	file1 = open(filepath, "r")
	
	bin2node = np.load(os.path.join(temp_dir, "bin2node.npy"), allow_pickle=True).item()
	
	line = file1.readline()
	count = 0
	final = []
	
	while line:
		info_list = line.strip().split("\t")[1:]
		temp = []
		
		for info in info_list:
			try:
				chrom, bin_ = info.split(":")
			except:
				print(info)
				raise EOFError
			if chrom not in chrom_list:
				continue
			bin_ = int(math.floor(int(bin_) / res)) * res
			bin_ = "%s:%d" % (chrom, bin_)
			node = bin2node[bin_]
			temp.append(node)
		temp = list(set(temp))
		
		temp.sort()
		count += 1
		if count % 100 == 0:
			print("%d\r" % count, end="")
			sys.stdout.flush()
		if len(temp) > 1:
			final.append(temp)
		
		line = file1.readline()
	
	return final
	
def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		id = int(np.argmax(memory_available))
		print("setting to gpu:%d" % id)
		torch.cuda.set_device(id)
		return "cuda:%d" % id
	else:
		return


def predict(model, input):
	model.eval()
	output = []
	new_batch_size = int(1e4)
	with torch.no_grad():
		for j in trange(math.ceil(len(input) / new_batch_size)):
			x = input[j * new_batch_size:min((j + 1) * new_batch_size, len(input))]
			x = np2tensor_hyper(x, dtype=torch.long)
			x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
			output.append(model(x).detach().cpu().numpy())
	torch.cuda.empty_cache()
	output = np.concatenate(output, axis=0)
	
	return output


if torch.cuda.is_available():
	current_device = get_free_gpu()
else:
	current_device = 'cpu'
	
	
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_dir = config['temp_dir']
res = config['resolution']
chrom_list = config['chrom_list']


args = parse_args()

if type(args.file) != str:
	print ("invalid filepath")
	raise EOFError
else:
	samples = np.array(parse_file(args.file))
	
classifier_model = torch.load(os.path.join(temp_dir, "model2load"), map_location=current_device)
proba = predict(classifier_model, samples)
proba = torch.sigmoid(torch.from_numpy(proba)).numpy()
np.savetxt(args.output, proba)