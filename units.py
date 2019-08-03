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


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco


def get_queryContent_label(pro_dataset, data_dir, outfile, test = True):
	if test:
		print ("start build final test file!!")
	else:
		print ("start build final train file!!")
	# data_dir = "data_set/clueweb09B-title/train_data.txt"

	# query_dir = "../data_set/clueweb09B-title/clueweb.title.krovetz.txt"
	query_dir = "../pre-process/"+pro_dataset+"/clueweb.title.krovetz.txt"
	
	# label_file = "../qrels_1-150.txt"
	label_file = "../pre-process/"+pro_dataset+"/qrels_1-150.txt"
	# outfile = "data_set/clueweb09B-title/corpus_query1_50.txt"
	f1 = open(data_dir)
	f2 = open(query_dir)
	f3 = open(label_file)
	# out_list = []

	query_dic = {}
	for query in f2.readlines():
		content = ""
		lineList = query.split("\t")
		for i in range(1,len(lineList)):
			content += lineList[i].strip("\n") + " "
		content = content.strip(" ")
		query_dic.update({lineList[0]:{}})
		query_dic[lineList[0]] = content
	# print (query_dic)

	label_dic = {}
	for label in f3.readlines():
		labelList = label.split(" ")
		if int(labelList[0]) <= 50:
			label_dic.update({labelList[0]+"_"+labelList[1]:labelList[2]})
		else:
			label_dic.update({labelList[0]+"_"+labelList[2]:labelList[3].strip()})
	# print(label_dic)
	# exit()
	out = open(outfile,'w')
	
	if test:		
		for line in f1.readlines():
			lineList = line.split("\t")
			if lineList[0]+"_"+lineList[1] in label_dic:
				out.write(lineList[0]+"\t"+query_dic[lineList[0]]+"\t"+lineList[1]+"\t"+lineList[2].strip()+"\t"+label_dic[lineList[0]+"_"+lineList[1]]+"\t"+lineList[3].strip()+"\n")
			else:
				out.write(lineList[0]+"\t"+query_dic[lineList[0]]+"\t"+lineList[1]+"\t"+lineList[2].strip()+"\t"+"0"+"\t"+lineList[3].strip()+"\n")
	else:
		
		for line in f1.readlines():
			lineList = line.split("\t")
			if lineList[0]+"_"+lineList[1] in label_dic:
				out.write(lineList[0]+"\t"+query_dic[lineList[0]]+"\t"+lineList[1]+"\t"+lineList[2].strip()+"\t"+label_dic[lineList[0]+"_"+lineList[1]]+"\t"+lineList[3].strip()+"\n")
			
	out.close()
	f3.close()
	f2.close()
	f1.close()
	#打乱顺序
	if test:
		data_file = os.path.join(outfile)
		data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
		counter = data.groupby("query_ID").apply(lambda data: sum(data["query_ID"] <= 150))
		query_is_qid = counter[counter>0].index
		df = data[data["query_ID"].isin(query_is_qid)]
		train_data,test_data = train_test_split(df,test_size=0.0)
		train_data.to_csv(outfile, sep='\t', header=False, index=False)
	else:
		data_file = os.path.join(outfile)
		data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
		counter = data.groupby("query_ID").apply(lambda data: sum(data["query_ID"] <= 150))
		query_is_qid = counter[counter>0].index
		df = data[data["query_ID"].isin(query_is_qid)]
		test_data,train_data = train_test_split(df,test_size=0)
		test_data.to_csv(outfile, sep='\t', header=False, index=False)
	if test:
		print ("end build final test file!!")
	else:
		print ("end build final train file!!")


def load(pro_dataset,file_name):
	print ("start load train and test datas!!")

	get_queryContent_label(pro_dataset, "../pre-process/" + pro_dataset+"/" + file_name + "/test_file.txt","../pre-process/" + pro_dataset + "/" + file_name + "/final_test_file.txt")
	get_queryContent_label(pro_dataset, "../pre-process/" + pro_dataset+"/" + file_name + "/train_file.txt","../pre-process/" + pro_dataset + "/" + file_name + "/final_train_file.txt", False)
	data_train = "../pre-process/" + pro_dataset + "/" + file_name + "/final_train_file.txt"
	data_test = "../pre-process/" + pro_dataset + "/" + file_name + "/final_test_file.txt"

	datas = []
	data_file = os.path.join(data_train)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	
	datas.append(data)
	data_file = os.path.join(data_test)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	datas.append(data)
	print ("end load train and test datas!!")
	return tuple(datas)


def read_trainSet(file,file1):
	inputFile = open(file)
	inputfile2 = open(file1)
	outputdic = {}
	qid = 0
	for line in inputFile.readlines():
		lineList = line.split('\t')
		docID = lineList[2]
		outputdic.update({lineList[0]+"_"+docID:[]})		
		word_count = lineList[3].strip()		
		word_count = word_count.split(" ")
		
		for word in word_count:
			outputdic[lineList[0]+"_"+docID].append(word)

	for line1 in inputfile2.readlines():
		lineList = line1.split('\t')
		docID = lineList[2]
		outputdic.update({lineList[0]+"_"+docID:[]})		
		word_count = lineList[3].strip()
		
		word_count = word_count.split(" ")
		
		for word in word_count:
			outputdic[lineList[0]+"_"+docID].append(word)

	print ("read_trainSet over!!")
	print ("\n")
	return outputdic


def get_word_dic(doc_wordList):
	wid = 0
	word_dic = {}
	for doc in doc_wordList:
		for item in doc_wordList[doc]:
			if item not in word_dic: 
				word_dic.update({item:wid})
				wid += 1
	print ("\nget_word_dic over!!\n")
	return word_dic


def add_query_dic(pro_dataset, word_dic, is_need=False):
	query_file = "../pre-process/"+pro_dataset+"/clueweb.title.krovetz.txt"
	file1 = open(query_file)
	query_word = []
	max_qid = 0

	# get max_index of word
	for index in word_dic:
		if word_dic[index]>max_qid:
			max_qid = word_dic[index]

	# build query_word dic
	for line in file1.readlines():
		linelist = line.strip("\n").split("\t")
		if int(linelist[0])==100:
			continue
		for word in linelist[1:]:
			if word not in query_word:
				query_word.append(word)
	max_qid += 1
	for item in query_word:
		if item not in word_dic:
			word_dic.update({item:max_qid})
			max_qid += 1
	file1.close()
	

def load_text_vec(word_dic,filename="",embedding_size = 100):
    vectors = {}
    with open(filename) as f:
        i = 0
        for line in f:
            i += 1
            if i % 100000 == 0:
                print ('epch %d' % i)
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print ( vocab_size, embedding_size)
            else:
                word = items[0]
                if word in word_dic:
                    vectors[word] = items[1:]
    # print ('embedding_size',embedding_size)
    # print ('done')
    # print ('words found in wor2vec embedding ',len(vectors.keys()))
    # print ("\nget words word2vec over!!\n")
    return vectors


def getSubVectorsFromDict(vectors,vocab,dim = 300):
    embedding = np.zeros((len(vocab),dim))
    print(len(vocab))
    count = 1
    temp_vec = 0
    for word in vocab:
        if word in vectors:
            count += 1
            embedding[vocab[word]]= np.array(vectors[word])      
        else:
            embedding[vocab[word]]= np.random.uniform(-0.25,+0.25,dim)#vectors['[UNKNOW]'] #.tolist()

        sum_ = 0.0
        for i in embedding[vocab[word]]:
        	sum_ += i*i

        sum_ = np.sqrt(sum_)
        for i,k in enumerate(embedding[vocab[word]]):
        	embedding[vocab[word]][i] /= sum_
	
    print ('word in embedding',count)
    return embedding


def get_wordDic_Embedding(pro_dataset,file_name, dim = 50):
	build_doc_word_dic = read_trainSet("../pre-process/"+pro_dataset+"/"+file_name+"/final_train_file.txt","../pre-process/"+pro_dataset+"/"+file_name+"/final_test_file.txt")
	word_dic = get_word_dic(build_doc_word_dic)
	add_query_dic(pro_dataset, word_dic, True)
	
	if dim == 50:
		fname = "../embedding/glove.6B/glove.6B.50d.txt"
		embeddings = load_text_vec(word_dic,fname,embedding_size = dim)
		sub_embeddings = getSubVectorsFromDict(embeddings,word_dic,dim)
	elif dim == 100:
		fname = "../embedding/glove.6B/glove.6B.100d.txt"
		embeddings = load_text_vec(word_dic,fname,embedding_size = dim)
		sub_embeddings = getSubVectorsFromDict(embeddings,word_dic,dim)
	elif dim == 200:
		fname = "../embedding/glove.6B/glove.6B.200d.txt"
		embeddings = load_text_vec(word_dic,fname,embedding_size = dim)
		sub_embeddings = getSubVectorsFromDict(embeddings,word_dic,dim)
	else:
		fname = '../embedding/glove.6B/glove.6B.300d.txt'
		embeddings = load_text_vec(word_dic,fname,embedding_size = dim)
		sub_embeddings = getSubVectorsFromDict(embeddings,word_dic,dim)
	print ("get wordDic and Embedding over!!")
	return word_dic,sub_embeddings

def cut(sentence):
    tokens = sentence.lower().split()
    return tokens

def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices = []    
    tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    while(len(indices) < max_sentence):
        indices += indices[:(max_sentence - len(indices))]
    return indices[:max_sentence]

# calculate the overlap_index
def overlap_index(question,answer,q_len,d_len,stopwords = []):
    qset = set(cut(question))
    aset = set(cut(answer))

    q_index = np.zeros(q_len)
    a_index = np.zeros(d_len)

    overlap = qset.intersection(aset)
    for i,q in enumerate(cut(question)[:q_len]):
        value = 1
        if q in overlap:
            value = 2
        q_index[i] = value
    for i,a in enumerate(cut(answer)[:d_len]):
        value = 1
        if a in overlap:
            value = 2
        a_index[i] = value
    return q_index,a_index


def get_tfidf_value(tfidf_value,max_qlen = 4,max_dlen = 50):
	return_list =[]
	zeros_list = []
	sum_value = 0
	tokens = cut(tfidf_value)
	
	for i in tokens:
		return_list.append(float(i))
		
	for i in range(max_dlen - len(return_list)):
		zeros_list.append(0)

	while(len(return_list) < max_dlen):
		return_list += zeros_list[:max_dlen - len(return_list)]

	sum_return_list = sum(return_list)

	for index,item in enumerate(return_list):
		return_list[index] = item/(sum_return_list*max_qlen)	

	return np.array(return_list[:max_dlen])


# @log_time_delta
# def batch_gen_with_test(df,alphabet,batch_size = 10,q_len = 33,d_len = 40,overlap_dict = None):
# 	pairs=[]
# 	input_num = 5
# 	for index,row in df.iterrows():
# 		query = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
# 		document = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
# 		word_tfidf = get_tfidf_value(row["tfidf"],max_qlen = q_len,max_dlen = d_len)

# 		if overlap_dict == None:
# 			q_overlap,d_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
# 		else:
# 			q_overlap,d_overlap = overlap_dict[(row["query_content"],row["document_content"])]

# 		pairs.append((query, document,q_overlap,d_overlap,word_tfidf))
		
# 	n_batches = int(len(pairs)*1.0 / batch_size)
# 	for i in range(0,n_batches):
# 		batch = pairs[i*batch_size:(i+1) * batch_size]
# 		yield [[pair[j] for pair in batch]  for j in range(input_num)]
	
# 	if (len(pairs)*1.0 % batch_size) == 0:
# 		batch =[pairs[n_batches*batch_size-1]] * (batch_size-len(pairs)+n_batches*batch_size)
# 		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]
# 	else:
# 		batch = pairs[n_batches*batch_size:]+[pairs[n_batches*batch_size]] * (batch_size-len(pairs)+n_batches*batch_size)
# 		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]


def transform(flag):
	return np.array([float(flag)], dtype = float)

@log_time_delta
def batch_gen_with_list_wise(df,alphabet, batch_size = 10,q_len = 4, d_len = 50,overlap_dict = None):
    #inputq inputa intput_y overlap
    input_num = 6
    dic = {}
    no_useList = []
   
    for index,row in df.iterrows():
        qid = row["query_ID"]
        question = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
        document = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
        word_tfidf = get_tfidf_value(row["tfidf"],max_qlen = q_len,max_dlen = d_len)

        label = transform(row["flag"])

        if overlap_dict == None:
            q_overlap,d_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
        else:
            q_overlap,d_overlap = overlap_dict[(row["query_content"],row["document_content"])]
        
        if qid in dic:
            dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))
        else:
            dic.update({qid:[]})
            dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))
        	
    for qid in dic:       
        pairs = dic[qid]               
        shuffle(pairs)       
        n_batches = int(len(pairs)*1.0 / batch_size)
        for i in range(0,n_batches):
            batch = pairs[i*batch_size:(i+1) * batch_size]
            yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]


@log_time_delta
def get_overlap_dict(df,alphabet,q_len = 40,d_len = 40):
    d = dict()
    for query in df['query_content'].unique():
        group = df[df['query_content'] == query]
        document_content = group['document_content']
        for doc in document_content:
            q_overlap,d_overlap = overlap_index(query,doc,q_len,d_len)
            d[(query,doc)] = (q_overlap,d_overlap)
    return d



@log_time_delta
def batch_gen_with_pair_wise(df,alphabet, batch_size = 10,q_len = 4, d_len = 50,overlap_dict = None):
	#inputq inputa intput_y overlap
    input_num = 6

    dic = {}

    pos_dic = {}
    neg_dic = {}
    no_useList = []
   	
    for index,row in df.iterrows():
        qid = row["query_ID"]
        question = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
        document = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
        word_tfidf = get_tfidf_value(row["tfidf"],max_qlen = q_len,max_dlen = d_len)

        label = transform(row["flag"])


        if overlap_dict == None:
            q_overlap,d_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
        else:
            q_overlap,d_overlap = overlap_dict[(row["query_content"],row["document_content"])]
      

        if label[0] >0:
            if qid in pos_dic:
                pos_dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))
            else:
                pos_dic.update({qid:[]})
                pos_dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))
        else:
            if qid in neg_dic:
                neg_dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))
            else:
                neg_dic.update({qid:[]})
                neg_dic[qid].append((question,document,label,q_overlap,d_overlap,word_tfidf))

    for qid in neg_dic:
        # if qid have no pos item?
        if qid not in pos_dic:
            continue        
        
        qid_neg_datas = neg_dic[qid]
        qid_pos_datas = pos_dic[qid]
        shuffle(qid_neg_datas)

        n_batches = int(len(qid_neg_datas)*1.0 / batch_size)
        for i in range(0,n_batches):
            pos_batch = []
            if len(qid_pos_datas)>=batch_size:
                pos_index = np.random.choice(len(qid_pos_datas), batch_size, False)
            else:
                shuffle(qid_pos_datas)
                pos_index = np.random.choice(len(qid_pos_datas), len(qid_pos_datas), False)
                for k in range(batch_size-len(qid_pos_datas)):
                    temp_index = np.random.choice(len(qid_pos_datas), 1, False)
                    pos_index = np.hstack((pos_index, pos_index[temp_index]))
           
            for index in pos_index:
                pos_batch.append(qid_pos_datas[index])            
            
            neg_batch = qid_neg_datas[i*batch_size:(i+1) * batch_size]
            
            return_pos_batch = [np.array([pair[j] for pair in pos_batch]) for j in range(input_num)]
            return_neg_batch = [np.array([pair[j] for pair in neg_batch]) for j in range(input_num)]
            yield [return_pos_batch, return_neg_batch]

@log_time_delta
def batch_gen_with_test(df,alphabet,batch_size = 10,q_len = 33,d_len = 40,overlap_dict = None):
	pairs=[]
	input_num = 5
	for index,row in df.iterrows():
		query = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
		document = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
		word_tfidf = get_tfidf_value(row["tfidf"],max_qlen = q_len,max_dlen = d_len)

		if overlap_dict == None:
			q_overlap,d_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
		else:
			q_overlap,d_overlap = overlap_dict[(row["query_content"],row["document_content"])]

		pairs.append((query, document,q_overlap,d_overlap,word_tfidf))
		
	n_batches = int(len(pairs)*1.0 / batch_size)
	for i in range(0,n_batches):
		batch = pairs[i*batch_size:(i+1) * batch_size]
		yield [[pair[j] for pair in batch]  for j in range(input_num)]
	
	if (len(pairs)*1.0 % batch_size) == 0:
		batch =[pairs[n_batches*batch_size-1]] * (batch_size-len(pairs)+n_batches*batch_size)
		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]
	else:
		batch = pairs[n_batches*batch_size:]+[pairs[n_batches*batch_size]] * (batch_size-len(pairs)+n_batches*batch_size)
		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]



