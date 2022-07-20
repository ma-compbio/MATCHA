import torch.nn as nn
import torch.nn.functional as F
from utils import *
import scipy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
activation = torch.tanh


def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
	''' For masking out the padding part of key sequence. '''
	
	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(0)
	padding_mask = padding_mask.unsqueeze(
		1).expand(-1, len_q, -1)  # b x lq x lk
	
	return padding_mask


class Wrap_Embedding(torch.nn.Embedding):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, *input):
		return super().forward(*input), torch.Tensor([0]).to(device)


# Used only for really big adjacency matrix
class SparseEmbedding(nn.Module):
	def __init__(self, embedding_weight, sparse=False):
		super().__init__()
		print(embedding_weight.shape)
		self.sparse = sparse
		if self.sparse:
			self.embedding = embedding_weight
		else:
			try:
				try:
					self.embedding = torch.from_numpy(
						np.asarray(embedding_weight.todense())).to(device)
				except BaseException:
					self.embedding = torch.from_numpy(
						np.asarray(embedding_weight)).to(device)
			except Exception as e:
				print("Sparse Embedding Error", e)
				self.sparse = True
				self.embedding = embedding_weight
	
	def forward(self, x):
		
		if self.sparse:
			x = x.cpu().numpy()
			x = x.reshape((-1))
			temp = np.asarray((self.embedding[x, :]).todense())
			
			return torch.from_numpy(temp).to(device)
		else:
			return self.embedding[x, :]


class TiedAutoEncoder(nn.Module):
	def __init__(self, shape_list, use_bias=True):
		super().__init__()
		self.weight_list = []
		self.bias_list = []
		self.use_bias = use_bias
		self.recon_bias_list = []
		for i in range(len(shape_list) - 1):
			self.weight_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i + 1], shape_list[i]).to(device)))
			self.bias_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i + 1]).to(device)))
			self.recon_bias_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i]).to(device)))
		self.recon_bias_list = self.recon_bias_list[::-1]
		
		for i, w in enumerate(self.weight_list):
			self.register_parameter('tied weight_%d' % i, w)
			self.register_parameter('tied bias1', self.bias_list[i])
			self.register_parameter('tied bias2', self.recon_bias_list[i])
		
		self.reset_parameters()
	
	def reset_parameters(self):
		for i, w in enumerate(self.weight_list):
			torch.nn.init.kaiming_uniform_(self.weight_list[i], a=math.sqrt(5))
		
		for i, b in enumerate(self.bias_list):
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_list[i])
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias_list[i], -bound, bound)
		temp_weight_list = self.weight_list[::-1]
		for i, b in enumerate(self.recon_bias_list):
			fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight_list[i])
			bound = 1 / math.sqrt(fan_out)
			torch.nn.init.uniform_(self.recon_bias_list[i], -bound, bound)
	
	def forward(self, input):
		# return input, input
		encoded_feats = input
		for i in range(len(self.weight_list)):
			if self.use_bias:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i], self.bias_list[i])
			else:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i])
			if i < len(self.weight_list) - 1:
				encoded_feats = activation(encoded_feats)
		
		reverse_weight_list = self.weight_list[::-1]
		reconstructed_output = encoded_feats
		for i in range(len(self.recon_bias_list)):
			reconstructed_output = F.linear(reconstructed_output, reverse_weight_list[i].t(), self.recon_bias_list[i])
			if i < len(self.recon_bias_list) - 1:
				reconstructed_output = activation(reconstructed_output)
		
		return encoded_feats, reconstructed_output


class MultipleEmbedding(nn.Module):
	def __init__(
			self,
			embedding_weights,
			dim,
			sparse=True,
			num_list=None,
			chrom_range=None,
			inter_initial=None):
		super().__init__()
		print(dim)
		self.chrom_range = chrom_range
		print("chrom_range", chrom_range)
		self.num_list = torch.tensor([0] + list(num_list)).to(device)
		print(self.num_list)
		self.dim = dim
		
		self.embeddings = []
		for i, w in enumerate(embedding_weights):
			self.embeddings.append(SparseEmbedding(w, sparse))
		
		if inter_initial is not None:
			for i in trange(len(inter_initial)):
				temp = inter_initial[i, :]
				inter_initial[i, temp > 0] = scipy.stats.mstats.zscore(temp[temp > 0]).astype('float32')
			
			# inter_initial[inter_initial > 0] = scipy.stats.mstats.zscore(inter_initial[inter_initial > 0], axis=1).astype('float32')
			inter_initial[np.isnan(inter_initial)] = 0.0
			
			self.inter_initial = SparseEmbedding(inter_initial, sparse)
		else:
			self.inter_initial = SparseEmbedding(w, sparse)
		
		test = torch.zeros(1, device=device).long()
		self.input_size = []
		for w in self.embeddings:
			self.input_size.append(w(test).shape[-1])
		
		self.wstack = [TiedAutoEncoder([self.input_size[i], self.dim, self.dim], use_bias=False).to(device) for i, w in
		               enumerate(self.embeddings)]
		self.next_w = FeedForward([self.dim, self.dim]).to(device)
		self.recon = [FeedForward([self.dim, v[1] - v[0]]).to(device) for i, v in enumerate(self.chrom_range)]
		
		for i, w in enumerate(self.wstack):
			self.add_module("Embedding_Linear%d" % (i), w)
			# self.add_module("Embedding_Linear", self.next_w)
			self.add_module("Embedding_recon%d" % (i), self.recon[i])
			
		
		self.dropout = nn.Dropout(0.2)
	
	def forward(self, x):
		
		final = torch.zeros((len(x), self.dim)).to(device)
		recon_loss = torch.Tensor([0.0]).to(device)
		for i in range(len(self.num_list) - 1):
			select = (x >= (self.num_list[i] + 1)) & (x < (self.num_list[i + 1] + 1))
			if torch.sum(select) == 0:
				continue
			adj = self.embeddings[i](x[select] - self.num_list[i] - 1)
			# output = adj
			output = self.dropout(adj)
			output, recon = self.wstack[i](output)
			final[select] = output
		
		final = final
		
		random_chrom = np.random.choice(np.arange(len(self.chrom_range)), 1)[0]
		# Get the bins in the other chromosome, and it cannot be 0 (because 0 is padding)
		other_chrom = ((x < self.num_list[random_chrom] + 1) | (x >= self.num_list[random_chrom + 1] + 1)) & (x != 0)
		if torch.sum(other_chrom) != 0:
			target = self.inter_initial(x[other_chrom] - 1)
			target = target[:, self.num_list[random_chrom]:self.num_list[random_chrom + 1]]
			recon = self.recon[random_chrom](activation(final[other_chrom]))
			recon_loss += (target - recon).pow(2).mean(dim=-1).mean() * 100
		
		return final, recon_loss


class Classifier(nn.Module):
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			node_embedding,
			diag_mask,
			bottle_neck,
			attribute_dict=None,
			**args):
		super().__init__()
		
		self.pff_classifier = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
		
		self.node_embedding = node_embedding
		self.encode1 = EncoderLayer(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul=0.3,
			dropout_pff=0.4,
			diag_mask=diag_mask,
			bottle_neck=bottle_neck)
		self.encode2 = EncoderLayer(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul=0.3,
			dropout_pff=0.4,
			diag_mask=diag_mask,
			bottle_neck=bottle_neck)
		self.diag_mask_flag = diag_mask
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.next_w = FeedForward([bottle_neck, bottle_neck]).to(device)
		if attribute_dict is not None:
			self.attribute_dict = torch.from_numpy(attribute_dict).to(device)
			self.attribute_dict_embedding = nn.Embedding(len(self.attribute_dict),1 ,padding_idx=0)
			self.attribute_dict_embedding.weight = nn.Parameter(self.attribute_dict)
			self.attribute_dict_embedding.weight.requires_grad = False
			self.attribute_nn = nn.Linear(self.attribute_dict.shape[-1], bottle_neck)
			self.attribute_dict = self.attribute_dict_embedding

	
	def get_node_embeddings(self, x, return_recon=False):
		# shape of x: (b, tuple)
		sz_b, len_seq = x.shape
		x, recon_loss = self.node_embedding(x.view(-1))
		if return_recon:
			return x.view(sz_b, len_seq, -1), recon_loss
		else:
			return x.view(sz_b, len_seq, -1)
	
	def get_embedding(self, x, slf_attn_mask, non_pad_mask, return_recon=False):
		sz_b, len_seq = x.shape
		attribute_embed = self.attribute_dict(x.view(-1))
		attribute_embed = self.attribute_nn(attribute_embed).view(sz_b, len_seq, -1)
		if return_recon:
			x, recon_loss = self.get_node_embeddings(x, return_recon)
		else:
			x = self.get_node_embeddings(x, return_recon)
		x += attribute_embed
		x = activation(self.next_w(x))
		dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
		# dynamic, static1, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
		if return_recon:
			return dynamic, static, attn, recon_loss
		else:
			return dynamic, static, attn
	
	def forward(self, x, mask=None, get_outlier=None, return_recon=False):
		x = x.long()
		
		slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
		non_pad_mask = get_non_pad_mask(x)
		
		# output, recon_loss = self.get_node_embeddings(x,return_recon=True)
		# output = output.view(len(output),1,-1)
		if return_recon:
			dynamic, static, attn, recon_loss = self.get_embedding(x, slf_attn_mask, non_pad_mask, return_recon)
		else:
			dynamic, static, attn = self.get_embedding(x, slf_attn_mask, non_pad_mask, return_recon)
		dynamic = self.layer_norm1(dynamic)
		static = self.layer_norm2(static)
		sz_b, len_seq, dim = dynamic.shape
		
		if self.diag_mask_flag:
			output = (dynamic - static) ** 2
		# output = dynamic * static
		else:
			output = dynamic
		output = self.pff_classifier(output)
		
		mode = 'sum'
		
		if mode == 'min':
			output, _ = torch.max(
				(1 - output) * non_pad_mask, dim=-2, keepdim=False)
			output = 1 - output
		
		elif mode == 'sum':
			output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
			mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False) + 1e-15
			output /= mask_sum
		elif mode == 'first':
			output = output[:, 0, :]
		# output = F.softplus(output)
		if return_recon:
			return output, recon_loss
		else:
			return output


# A custom position-wise MLP.
# dims is a list, it would create multiple layer with tanh between them
# If dropout, it would add the dropout at the end. Before residual and
# layer-norm


class PositionwiseFeedForward(nn.Module):
	def __init__(
			self,
			dims,
			dropout=None,
			reshape=False,
			use_bias=True,
			residual=False,
			layer_norm=False):
		super(PositionwiseFeedForward, self).__init__()
		self.w_stack = []
		self.dims = dims
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias))
			self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
		self.reshape = reshape
		self.layer_norm = nn.LayerNorm(dims[-1])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.residual = residual
		self.layer_norm_flag = layer_norm
	
	def forward(self, x):
		output = x.transpose(1, 2)
		
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation(output)
			if self.dropout is not None:
				output = self.dropout(output)
		
		output = self.w_stack[-1](output)
		output = output.transpose(1, 2)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		if self.dims[0] == self.dims[-1]:
			# residual
			if self.residual:
				output += x
			
			if self.layer_norm_flag:
				output = self.layer_norm(output)
		
		return output


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier


class FeedForward(nn.Module):
	''' A two-feed-forward-layer module '''
	
	def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
		super(FeedForward, self).__init__()
		self.w_stack = []
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
			self.add_module("FF_Linear%d" % (i), self.w_stack[-1])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.reshape = reshape
	
	def forward(self, x):
		output = x
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation(output)
			if self.dropout is not None:
				output = self.dropout(output)
		output = self.w_stack[-1](output)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		return output


class ScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product Attention '''
	
	def __init__(self, temperature):
		super().__init__()
		self.temperature = temperature
	
	def masked_softmax(self, vector: torch.Tensor,
	                   mask: torch.Tensor,
	                   dim: int = -1,
	                   memory_efficient: bool = False,
	                   mask_fill_value: float = -1e32) -> torch.Tensor:
		
		if mask is None:
			result = torch.nn.functional.softmax(vector, dim=dim)
		else:
			mask = mask.float()
			while mask.dim() < vector.dim():
				mask = mask.unsqueeze(1)
			if not memory_efficient:
				# To limit numerical errors from large vector elements outside
				# the mask, we zero these out.
				result = torch.nn.functional.softmax(vector * mask, dim=dim)
				result = result * mask
				result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
			else:
				masked_vector = vector.masked_fill(
					(1 - mask).bool(), mask_fill_value)
				result = torch.nn.functional.softmax(masked_vector, dim=dim)
		return result
	
	def forward(self, q, k, v, diag_mask, mask=None):
		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature
		
		if mask is not None:
			attn = attn.masked_fill(mask, -float('inf'))
		
		attn = self.masked_softmax(
			attn, diag_mask, dim=-1, memory_efficient=True)
		
		output = torch.bmm(attn, v)
		
		return output, attn


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout,
			diag_mask,
			input_dim):
		super().__init__()
		
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
		
		nn.init.normal_(self.w_qs.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0,
		                std=np.sqrt(2.0 / (d_model + d_v)))
		
		self.attention = ScaledDotProductAttention(
			temperature=np.power(d_k, 0.5))
		
		self.fc1 = nn.Linear(n_head * d_v, d_model)
		self.fc2 = nn.Linear(n_head * d_v, d_model)
		
		self.layer_norm1 = nn.LayerNorm(input_dim)
		self.layer_norm2 = nn.LayerNorm(input_dim)
		self.layer_norm3 = nn.LayerNorm(input_dim)
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = dropout
		
		self.diag_mask_flag = diag_mask
		self.diag_mask = None
	
	def pass_(self, inputs):
		return inputs
	
	def forward(self, q, k, v, diag_mask, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		
		residual_dynamic = q
		residual_static = v
		
		q = self.layer_norm1(q)
		k = self.layer_norm2(k)
		v = self.layer_norm3(v)
		
		sz_b, len_q, _ = q.shape
		sz_b, len_k, _ = k.shape
		sz_b, len_v, _ = v.shape
		
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		
		q = q.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_q, d_k)  # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_k, d_k)  # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_v, d_v)  # (n*b) x lv x dv
		
		n = sz_b * n_head
		
		if self.diag_mask is not None:
			if (len(self.diag_mask) <= n) or (
					self.diag_mask.shape[1] != len_v):
				self.diag_mask = torch.ones((len_v, len_v), device=device)
				if self.diag_mask_flag:
					self.diag_mask -= torch.eye(len_v, len_v, device=device)
				self.diag_mask = self.diag_mask.repeat(n, 1, 1)
				diag_mask = self.diag_mask
			else:
				diag_mask = self.diag_mask[:n]
		
		else:
			self.diag_mask = (torch.ones((len_v, len_v), device=device))
			if self.diag_mask_flag:
				self.diag_mask -= torch.eye(len_v, len_v, device=device)
			self.diag_mask = self.diag_mask.repeat(n, 1, 1)
			diag_mask = self.diag_mask
		
		if mask is not None:
			mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
		
		dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
		
		dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
		dynamic = dynamic.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		static = v.view(n_head, sz_b, len_q, d_v)
		static = static.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		
		dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
		static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
		
		return dynamic, static, attn


class EncoderLayer(nn.Module):
	'''A self-attention layer + 2 layered pff'''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul,
			dropout_pff,
			diag_mask,
			bottle_neck):
		super().__init__()
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.mul_head_attn = MultiHeadAttention(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout=dropout_mul,
			diag_mask=diag_mask,
			input_dim=bottle_neck)
		self.pff_n1 = PositionwiseFeedForward(
			[d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
		self.pff_n2 = PositionwiseFeedForward(
			[bottle_neck, d_model, d_model], dropout=dropout_pff, residual=False, layer_norm=True)
	
	# self.dropout = nn.Dropout(0.2)
	
	def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
		dynamic, static1, attn = self.mul_head_attn(
			dynamic, dynamic, static, slf_attn_mask)
		dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
		static1 = self.pff_n2(static * non_pad_mask) * non_pad_mask
		
		return dynamic, static, attn


class DataGenerator():
	def __init__(self, edges, edge_weight, batch_size, num_batch_per_iter, min_size = 2, max_size = 2, flag=False):
		self.edges = [[] for i in range(max_size + 1)]
		self.edge_weight = [[] for i in range(max_size + 1)]
		
		for e, ew in zip(edges, edge_weight):
			self.edges[len(e)].append(e)
			self.edge_weight[len(e)].append(ew)
		
		self.batch_size = batch_size
		self.num_batch_per_iter = num_batch_per_iter
		self.min_size = min_size
		self.max_size = max_size
		self.flag = flag
		for i in range(min_size, max_size + 1):
			self.edges[i] = np.array(self.edges[i])
			self.edge_weight[i] = np.array(self.edge_weight[i])
		
			while len(self.edges[i]) <= self.num_batch_per_iter * self.batch_size:
				self.edges[i] = np.concatenate([self.edges[i], self.edges[i]])
				self.edge_weight[i] = np.concatenate([self.edge_weight[i], self.edge_weight[i]])
			self.shuffle(i)
		self.pointer = np.zeros((len(self.edges)),dtype='int')

		
	
	def shuffle(self, i):
		if self.flag:
			print("reach end, shuffling")
		index = np.random.permutation(len(self.edges[i]))
		self.edges[i] = (self.edges[i])[index]
		self.edge_weight[i] = (self.edge_weight[i])[index]
	
	def next_iter(self):
		
		return_edges = []
		return_edge_weight = []
		
		for i in range(self.min_size, self.max_size + 1):
			self.pointer[i] += self.num_batch_per_iter * self.batch_size
			
			if self.pointer[i] <= len(self.edges[i]):
				index = range(self.pointer[i] - self.num_batch_per_iter * self.batch_size, min(self.pointer[i], len(self.edges[i])))
				edges = (self.edges[i])[index]
				edge_weight = (self.edge_weight[i])[index]
				return_edges+= list(edges)
				return_edge_weight += list(edge_weight)
			else:
				index = range(self.pointer[i] - self.num_batch_per_iter * self.batch_size,
				              min(self.pointer[i], len(self.edges[i])))
				edges = (self.edges[i])[index]
				edge_weight = (self.edge_weight[i])[index]
				
				self.shuffle(i)
				left = self.num_batch_per_iter * self.batch_size - len(index)
				self.pointer[i] = 0
				self.pointer[i] += left
				index = range(0, self.pointer[i])
				edges, edge_weight =  np.concatenate([edges, self.edges[i][index]]), np.concatenate([edge_weight, self.edge_weight[i][index]])
				return_edges += list(edges)
				return_edge_weight += list(edge_weight)
		return np.asarray(return_edges), np.asarray(return_edge_weight)
	
	def balance_num(self, edges):
		cell = edges[:, 0]
		final = []
		choice, counts_ = np.unique(cell, return_counts=True)
		# num = int(np.mean(counts_))
		num = 50
		for c in tqdm(choice):
			final.append(np.random.choice(np.where(cell == c)[0], num, replace=True))
		final = np.concatenate(final, axis=-1)
		return final
