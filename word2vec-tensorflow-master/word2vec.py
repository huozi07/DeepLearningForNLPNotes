from util import *
from train import *
from crawler import *
import os
import time
import pickle

def main():	
	args = parse_arguments()

	if args.sub == "crawl":
		lc = LawCrawler(args.domain, args.num_process, args.verbose)
		data = lc.start()

		with open(args.output, "w") as w:
			w.write("{0}" .format(data))
	
	if args.sub == "word2vec":	
		# To store result files
		if not os.path.exists("result/"):
			os.makedirs("result/")

		# Store/load preprocessed data
		try:
			with open("result/preprocess", "rb") as pre_data:
				print("Use preprocessd data.")
				word_dict = pickle.load(pre_data)
				word2idx = pickle.load(pre_data)
		except IOError as e:
			word2idx, word_dict = preprocess(args)
			with open("result/preprocess", "wb") as pre_data:
				pickle.dump(word_dict, pre_data)
				pickle.dump(word2idx, pre_data)

		start_time = time.time()
		result = train(word2idx, args)
		end_time = time.time()
		print("Train word2vec done. {0:.2f} sec." .format(end_time-start_time))

		# Store trained data (word vector).
		with open("result/"+args.result, "wb") as w2v_data:
			pickle.dump(result, w2v_data)

	if args.sub == "vis":
		try:
			with open("result/result", "rb") as w2v_data:
				result = pickle.load(w2v_data)

			with open("result/preprocess", "rb") as pre_data:
				word_dict = pickle.load(pre_data)
		except IOError as e:
			print("No result files.")
			log.close()
		
		start_time = time.time()
		visualization(result, word_dict)
		end_time = time.time()
		print("t-SNE and visualization done. {0:.2f} sec." .format(end_time-start_time))


if __name__ == "__main__":
	main()
