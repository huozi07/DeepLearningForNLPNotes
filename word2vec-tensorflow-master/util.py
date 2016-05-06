import argparse
import re
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from konlpy.tag import Mecab

def parse_arguments():
	parser = argparse.ArgumentParser()
	sub = parser.add_subparsers(dest="sub")

	parser_crawl = sub.add_parser("crawl", help="Crawl law data.")
	parser_w2v   = sub.add_parser("word2vec", help="Do word2vec. Before word2vec, preprocess law data and build batch.")
	parser_vis   = sub.add_parser("vis", help="Do t-SNE and visualization.")

	parser_crawl.add_argument("--num_process", type=int, default=1, metavar="",
							  help="Number of process.")
	parser_crawl.add_argument("--domain", type=str, default="lsSc", metavar="",
							  help="Base domain to parse.")
	parser_crawl.add_argument("--output", type=str, default="data/data.txt", metavar="",
							  help="Output filename.")
	parser_crawl.add_argument("--verbose", action="store_true", default=False,
							  help="Print progress during execution.")
	
	parser_w2v.add_argument("--input", type=str, default="data/data.txt", metavar="",
						    help="Input filename to preprocess (default=data/data.txt).")
	parser_w2v.add_argument("--voca_size", type=int, default="20000", metavar="",
						    help="Number of vocabulary size (default=20000).")
	parser_w2v.add_argument("--window", type=int, default="1", metavar="",
						    help="Maximum distance of target and predicted one (default=1).")
	parser_w2v.add_argument("--batch", type=int, default="10000", metavar="",
						    help="Number of minibatch size (default=10000).")
	parser_w2v.add_argument("--hidden_size", type=int, default="100", metavar="",
						    help="Number of neurons in hidden layer (default=100).")
	parser_w2v.add_argument("--neg_sample", type=int, default="128", metavar="",
						    help="Number of negative sample when calc NCE loss (default=128).")
	parser_w2v.add_argument("--num_iter", type=int, default="10000", metavar="",
						    help="Number of iterations to train (default=10000).")
	parser_w2v.add_argument("--lr", type=float, default="0.025", metavar="",
						    help="Learning rate (default=0.025).")
	parser_w2v.add_argument("--verbose", action="store_true", default=False,
							help="Print progress during execution.")
	parser_w2v.add_argument("--result", type=str, default="result", metavar="",
							help="Result filename")

	return parser.parse_args()
	
	
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	plt.figure(figsize=(60, 60))  #in inches
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		plt.scatter(x, y)
		plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
					 family=["NanumGothicCoding"])

	plt.savefig(filename)


def visualization(result, word_dict):
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	plot_only = 500

	low_dim_embs = tsne.fit_transform(result[0:500])
	labels = [ word_dict[i] for i in range(500) ]
	plot_with_labels(low_dim_embs, labels)


def preprocess(args):
	"""
	Description

	Return
	- word2idx: Sequence of word index. It is 2-dim like [# of laws, # of words in each law].
	- word_dict: Word to index mapping table. { word: idx } (Only contain VOCA_SIZE words)
	- word_inv_dict: Inverted version of word_dict. { idx: word } (Only contain VOCA_SIZE words)
	- word_count: Word counter of each laws. Only contain VOCA_SIZE words.
	"""
	tagger = Mecab()
	
	with open(args.input, "r") as reader:
		data = reader.read()

	# Sequence of words in each law. [num_laws, num_words]
	word_list     = list()
	# Sequence of idx. [num_laws, num_words]
	word2idx      = list()
	# Mapping table of word - idx.
	word_dict     = dict()
	# Inversed mapping table of word - idx (for fast access).
	word_inv_dict = dict()
	# Word counter.
	word_count    = list()

	""" Tag part-of-speech and remove unimportant words (like josa..). """
	# Split each laws by <END> symbol.
	law_list = data.split("<END>")
	for law in law_list:
		# Eliminate special chars
		law = re.sub("[^a-zA-Z0-9가-힣 \n]", " ", law)
		# 1. Eliminate newline, tab and strange char.
		# 2. Split words by space.
		word_list.append(law.replace("\n", " ").replace("\t", " ").replace("\xa0" ,"").split(" "))

	for i, v in enumerate(word_list):
		for j, word in enumerate(v):
			# Tag laws using Mecab tagger. and exclude some tags.
			tag = tagger.pos(word)
			excluded = [ t[0] for t in tag if not re.search("NN|XR", t[1]) ]
		
			# Exclude word if it contain number (ex. 제1조, 제1항의 경우 해당 단어 삭제).
			for t in tag:
				if t[1] == "SN": word_list[i][j] = ""
			
			# Reconstruct word_list by using excluded tag list.
			for e in excluded:
				word_list[i][j] = word_list[i][j].replace(e, "")

		word_list[i] = [ w for w in word_list[i] if len(w) > 1 or w == "법" ]
	
	# If last element of word_list is empty, remove it.
	if not word_list[-1]:
		word_list.pop()
	
	# Construct word counter. 1st element in counter is UNKOWN_WORD (simply UNK).
	word_count.append(["UNK", 0])
	merged = list(itertools.chain.from_iterable(word_list))
	word_count.extend(collections.Counter(merged).most_common(args.voca_size-1))

	# Construct word mapping table.
	word_dict = { v[0] : i for v, i in zip(word_count, itertools.count(0)) }
	word_inv_dict = { i : v for v, i in word_dict.items() }

	# Make sequence of word-idx.
	for v in word_list:
		row = list()
		for word in v:
			idx = word_dict.get(word)
			if idx != None: 
				row.append(idx)
			else: 			
				row.append(word_dict.get("UNK"))
				word_count[0][1] += 1
		word2idx.append(row)

	word_list = None # dont use anymore
	word_dict = None # dont use anymore
	word_count = None # dont use anympre
	return np.array(word2idx), word_inv_dict


def generate_batch(word2idx, args):
	"""
	Input

	Return
	batch: 
	label: 
	"""
	batch = np.zeros((args.batch), dtype=np.int32)
	label = np.zeros((args.batch, 1), dtype=np.int32)

	for i in range(args.batch):
		row_idx = np.random.choice(len(word2idx), 1)[0]
		target_idx = np.random.choice(len(word2idx[row_idx]), 1)[0]
		target = word2idx[row_idx][target_idx]

		border = [max(0, target_idx-args.window), 
				  min(len(word2idx[row_idx])-1, target_idx+args.window)]
		predict_idx = np.random.choice(border, 1)
		predict = word2idx[row_idx][predict_idx]
	
		batch[i] = target
		label[i] = predict

	"""
	Try to vectorize it..
	row_idx = np.random.randint(0, len(word2idx), args.batch)
	target_idx = np.random.randint(0, len(word2idx[row_idx]), args.batch)
	print(word2idx)
	target = word2idx[row_idx, target_idx]

	border = np.array([np.maximum(0, target_idx-args.window), 
					   np.minimum(len(word2idx[row_idx])-1, target_idx+args.window)]).T
	predict_idx = np.zeros([args.batch])
	for i in range(args.batch):
		predict_idx = np.random.randint(border[i][0], border[i][1], 1)
	predict = word2idx[row_idx, predict_idx]
		
	batch = target
	label = predict

	print(target)
	#print(label.shape)
	"""

	return batch, label	
