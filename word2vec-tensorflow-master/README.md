# word2vec implementation in tensorflow
This code first crawl **Korean Law data** in [here](http://mobile.law.go.kr), and do word2vec and visualization.<br>
Basically each laws is separated by symbol `<END>`. If your text has no `<END>` symbol (just a single document) it will be no problem too.

## Basic Usage
1. To crawl law data, use command `python word2vec.py crawl`.
2. `python word2vec.py word2vec` will excute preprocess of data and word2vec.<br>
You can also use `python script.py` that I made for test.
3. Visualization using t-SNE can perform with `python word2vec.py vis`.

For more information about arguemts, press `python word2vec [crawl|word2vec|vis] --help` or see `parse_argument` function in `util.py`.
