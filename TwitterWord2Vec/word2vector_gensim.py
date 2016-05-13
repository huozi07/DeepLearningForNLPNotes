#!/usr/bin/python
# -*- coding: utf-8 -*-
import gensim.models
import time
import pandas as pd
from nltk.tokenize import TweetTokenizer
time1 = time.time()
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def loaddata(inputfile):
    file = open(inputfile)
    tknzr = TweetTokenizer()
    sentences=[]
    while 1:
        line = file.readline().strip()
        if not line:
              break
        sentences.append(tknzr.tokenize(line))
    return sentences

def WordFrequencyAnalysi():
    #load data
    # sentences = [['first', 'sentence'], ['second', 'sentence']]
    sentences = gensim.models.word2vec.LineSentence("Tweets10")#导入词汇
    print sentences

    modelbase = gensim.models.Word2Vec(min_count=1)
    modelbase.build_vocab(sentences)#构建词汇哈夫曼树,节省训练时找词汇的搜索时间
    #根据词频分布\确定min_count 如取前1000个词汇
    wordCount=[]
    for i in  modelbase.vocab.keys():
          wordCount.append((i,modelbase.vocab[i].count))
    print wordCount

def trainModel(inputfile,outVectorFile):
    #load data
    sentences=loaddata(inputfile)
    modelbase = gensim.models.Word2Vec(min_count=1)
    modelbase.build_vocab(sentences)
    modelbase.train(sentences)#必须构建完词汇哈弗曼树才可以train model
    #model save
    modelbase.save_word2vec_format(outVectorFile)#存为词向量
    #model using

def transformVectorToGraphTable(outVectorFile,yuzhi):
    print "load and transform data"
    file = open(outVectorFile)
    Vectors=[]
    while 1:
        line = file.readline().strip()
        if not line:
              break
        if len(line.split(" "))!=101:print len(line.split(" ")),line
        Vectors.append(line.split(" "))
    matrix=np.matrix(Vectors[1:]).T
    Vectors=pd.DataFrame(matrix[1:],dtype='float64')
    Vectors.columns=matrix[0].tolist()[0]
    print "---Compute Euclidean Result---"
    #自定义实现欧几里得距离计算节点相似度
    distance = lambda column1, column2: pd.np.linalg.norm(column1-column2)
    EuclideanResult = Vectors.apply(lambda col1: Vectors.apply(lambda col2: distance(col1, col2)))
    print "---output graph edge-------"
    GraphEdge=[]

    index=1
    for idx,row in EuclideanResult.iterrows():
        for col in Vectors.columns[:index]:
                if  row[col]<yuzhi and row[col]!=float(0):
                       GraphEdge.append([idx,col,str(row[col])])
        index+=1
    GraphEdge=pd.DataFrame(GraphEdge,columns=["source","target","weight"])
    print "Edge Shape",GraphEdge.shape
    GraphEdge.to_csv("GraphEdge.csv")

if __name__=="__main__":
    inputfile="XXX.txt"
    outVectorFile="XXX"
    #词频分析确定参数 MiniCount
    WordFrequencyAnalysi()
    #train and save model
    trainModel(inputfile,outVectorFile)
    #Visulization of result
    transformVectorToGraphTable(outVectorFile,0.035)
