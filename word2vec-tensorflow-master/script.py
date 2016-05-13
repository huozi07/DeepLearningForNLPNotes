import subprocess
import numpy as np

start = 0.01
end = 0.1

for i in range(1):
	lr = 0.007
	todo = "python word2vec.py word2vec \
			--input data/data.txt \
			--result result \
			--voca_size 20000 \
			--batch 4096 \
			--hidden_size 512 \
			--num_iter 10000 \
			--window 5 \
			--verbose \
			--lr " + str(lr)
	subprocess.call(todo, shell=True)
