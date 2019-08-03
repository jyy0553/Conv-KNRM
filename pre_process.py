#coding=utf-8
import sys
import os
import json
import math
import copy

import pandas as pd
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.Snowball import SnowballStemmer 
# from gensim.models.keyedvectors import KeyedVectors
from random import *
import numpy as np
import importlib
import pickle
import sklearn
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import heapq
from sklearn.model_selection import train_test_split
from functools import wraps
import time
stoplist = stopwords.words('english')
#creat a list which contains query and its candidate document set
#each candidate document set have many document which
wlem = WordNetLemmatizer()


#划分文档集
def divide_document_features_Set(pro_dataset,outfile_name):
	print ("start divide document set to test and train!!")
	f = open("../pre-process/"+pro_dataset+"/document_features_query1_150_tfidf_top50.txt")
	out1 = open("../pre-process/"+pro_dataset+"/"+outfile_name+"/train_file.txt",'w')
	out2 = open("../pre-process/"+pro_dataset+"/"+outfile_name+"/test_file.txt",'w')
	qid_out3 = open("../pre-process/"+pro_dataset+"/"+outfile_name+"/readme.txt",'w') 
	qid = []
	for i in range(1,151):
		if i != 100 and i!=95 and i!= 20:
			qid.append(i)
	trainID,testID = train_test_split(qid,test_size=0.2)
	trainID.append(20)

	# record qid in train or test
	trainQid_write_str = "train qids : "
	for qid in trainID:
		trainQid_write_str+=str(qid)+", "
	qid_out3.write(trainQid_write_str+"\n")	
	testQid_write_str = "test qids : "
	for qid in testID:
		testQid_write_str+=str(qid)+", "
	qid_out3.write(testQid_write_str+"\n")	
	
	for line in f.readlines():
		words = ""
		tfidfs = []
		w_tfidf = ""
		lineList = line.split("\t")
		if int(lineList[0]) in trainID:
			items = lineList[2].strip()
			items = items.split(" ")
			for item in items:
				item = item.split(":")
				words += item[0]+" "
				tfidfs.append(float(item[1]))
			words = words.strip()
			#tfidf归一化
			sum_tfidf = sum(tfidfs)
			index = 0
			for tfidf in tfidfs:
				tfidfs[index] = tfidf/sum_tfidf
				w_tfidf+=str(tfidfs[index])+" "
				index+=1
			w_tfidf = w_tfidf.strip()		
			out1.write(lineList[0]+"\t"+lineList[1]+"\t"+words+"\t"+w_tfidf+"\n")
			

		elif int(lineList[0]) in testID:
			items = lineList[2].strip()
			items = items.split(" ")
			for item in items:
				item = item.split(":")
				words += item[0]+" "
				tfidfs.append(float(item[1]))
			words = words.strip()
			#tfidf归一化
			sum_tfidf = sum(tfidfs)
			index = 0
			for tfidf in tfidfs:
				tfidfs[index] = tfidf/sum_tfidf
				w_tfidf+=str(tfidfs[index])+" "
				index+=1
			w_tfidf = w_tfidf.strip()		
			out2.write(lineList[0]+"\t"+lineList[1]+"\t"+words+"\t"+w_tfidf+"\n")

	out2.close()
	out1.close()
	f.close()
	print ("end divide !!")


data_set = "clueweb09B-title"
for i in range(1,6):
	file_name = "File"+str(i)
	divide_document_features_Set(data_set, file_name)