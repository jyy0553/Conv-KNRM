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


# step 1
def read_dataset(file,start):
	print ("\nstart read dataset!!!\n")
	inputFile = open(file,'rb')
	outputdic = {}
	qid = 0
	j = 0
	for line in inputFile.readlines():
		j += 1
		# print (j)
		if j < start:
			continue
		lineList = str(line).split('\\t')
		docID = lineList[0].strip("b'")
		# print (lineList[1])
		outputdic.update({docID:{}})		
		word_count = lineList[2].strip("\\n'")
		word_count = word_count.split(' ')
		for word in word_count:
			word1 = word.split(":")
			if len(word1) != 2:
				continue
			# if int(word1[1])>2 :
			outputdic[docID].update({wlem.lemmatize(word1[0]):int(word1[1])})			
		# print (outputdic)	
		#total about 30w
		if j == 119999:
			break
		if j == 249999:
			break

		if j%10000 == 0:
			print (j)
	print ("\nread_doc_word over!!\n")
	inputFile.close()
	return outputdic
# step 3
def read_candidateDocumentSet(file):

	print ("\nstart read candidate DocumentSet list!!\n")
	inputFile = open(file)
	outputdic = {}
	i = 0
	j = 0
	for line in inputFile.readlines():
		lineList = str(line).split('\t')
		if int(lineList[3]) <= 1000:
			continue
		qid = int(lineList[0])
		if qid == i:
			outputdic[qid].append(lineList[2])
		else:
			i = qid
			outputdic.update({qid:[]})
			outputdic[qid].append(lineList[2])
		# j += 1
		# if j >=1500 :
		# 	break
	# print (i)
	
	inputFile.close()
	print ("\nread candidate DocumentSet list over!!\n")
	return outputdic

# step 2
def write_doc_content(doc_word_dic):
	print ("\nstart write doc content!!\n")
	filename = '../data_set/clueweb09B-title/doc_content.txt'
	with open(filename,'a') as f:
		for docID in doc_word_dic:
			doc_content = ""
			doc_content += docID+"\t"
			# print (docID)
			# print (doc_word_dic[docID])
			
			for word in doc_word_dic[docID]:
				for i in range(doc_word_dic[docID][word]):
					doc_content += word+" "
			# print (doc_content)
			f.write(doc_content+"\n")
			# break
	print ("\nwrite doc content File over!!\n")


# build_doc_word_dic = read_dataset("data_set/clueweb09B-title/clueweb.title.galago.docset",0)
# write_doc_content(build_doc_word_dic)
# build_doc_word_dic1 = read_dataset("data_set/clueweb09B-title/clueweb.title.galago.docset",120000)
# write_doc_content(build_doc_word_dic1)
# build_doc_word_dic2 = read_dataset("data_set/clueweb09B-title/clueweb.title.galago.docset",250000)
# write_doc_content(build_doc_word_dic2)

# write_doc_content()

# candidate_documentSet = read_candidateDocumentSet("data_set/clueweb09B-title/clueweb.title.galago.2k.out")
# print (candidate_documentSet)

# step 4
def read_candidate_doc_content(file,candidate_documentSet):
	print ("\nstart read candidate doc content!!!\n")	
	outputdic = {}
	i = 0
	# filename = 'data_set/clueweb09B-title/candidate_doc_content.txt'
	# filename = 'data_set/clueweb09B-title/candidate_doc_content_2000.txt'
	filename = '../data_set/clueweb09B-title/candidate_doc_content_1001_2000.txt'
	with open(filename,'w') as f:
		for query in candidate_documentSet:
			# if int(query) <=87:
			# 	continue
			# if int(query) <=8:
			# 	continue
			for docID in candidate_documentSet[query]:
				inputFile = open(file)
				for line in inputFile.readlines():
					lineList = str(line).split('\t')
					if lineList[0] == docID:
						content = lineList[1]
						f.write(str(query)+"\t"+docID+"\t"+content)
						inputFile.close()
						break
			# outputdic.update()
				
			# break	
	print ("\nread candidate doc content over!!!\n")

# document_features_query1_50.txt
# read_candidate_doc_content("data_set/clueweb09B-title/doc_content.txt",candidate_documentSet)
# read_candidate_doc_content("data_set/clueweb09B-title/doc_content.txt",candidate_documentSet)
# step 5
def compute_tfidf(inFile,outputFile,qid):
	f = open(outputFile,'a')
	corpus = []
	docID = []
	inputFile = open(inFile)
	isRun = 0
	for line in inputFile.readlines():
		lineList = str(line).split('\t')
		if int(lineList[0]) != int(qid) and isRun == 0:
			continue
		if int(lineList[0]) != int(qid) and isRun != 0:
			break
		isRun = 1
		corpus.append(lineList[2].strip("\n'"))
		docID.append(lineList[0]+"\t"+lineList[1]+"\t")
	vectorizer = CountVectorizer()
	transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
	tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵 
	word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
	weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
	for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
		f.write(docID[i])
		for j in range(len(word)):  
			if weight[i][j]>0.0:
				f.write(word[j]+":"+str(weight[i][j])+" ")
		f.write("\n")
	f.close()
	print ("over compute query %d tfidf!"%qid)
# for i in range(2,51):
	# compute_tfidf('data_set/clueweb09B-title/candidate_doc_content.txt',"data_set/clueweb09B-title/query_doc_tfidf.txt",i)

#step 6
def extract_document_features(inputFile,outputFile):
	i=0
	f = open(inputFile)
	outFile = open(outputFile,'w')
	for line in f.readlines():
		count = 0
		qid_docID = ""	
		sort_word_tfidf = ""
		word_tfidfDic = {}	
		lineList = line.split('\t')
		qid_docID+=lineList[0]+"\t"+lineList[1]+"\t"
		word_ = lineList[2].strip("\n")
		word_ = word_.strip()
		word = word_.split(' ')
		for item in word:
			itemList = item.split(":")
			word_tfidfDic.update({itemList[0]:float(itemList[1])})

		value_sort = sorted(word_tfidfDic.items(), key=lambda d:d[1], reverse = True)
		for value,key in value_sort:
			
			if count<=20 and value not in stoplist:
				count += 1
				sort_word_tfidf += value+":"+str(key)+" "
			elif count>20:
				sort_word_tfidf += "\n"
				break
		outFile.write(qid_docID+sort_word_tfidf)
		# print (qid_docID+sort_word_tfidf)
		outFile.close()
		# break
		# i+=1
		# if i>1000:
		# 	break

# extract_document_features("data_set/clueweb09B-title/query1_50_doc_tfidf.txt","data_set/clueweb09B-title/document_features_query1_50.txt")


def count_same_number(inputFile1,inputFile2,outputFile,qid):
	inputFile1 = open(inputFile1)
	inputFile2 = open(inputFile2)
	f = open(outputFile,"a")
	sum = 0
	# dic = []
	dic = {}
	for line in inputFile1.readlines():
		lineList = line.split('\t')
		if lineList[0] == qid:
			dic.update({lineList[2]:lineList[3]})
			# dic.append(lineList[2])
	print ("len:"+str(len(dic)))
	for line in inputFile2.readlines():
		lineList = line.split('\t')
		if lineList[0] == qid and lineList[1] in dic:
			# print (lineList[2])
			content = str(lineList[2].strip('\n'))
			content = content.strip(" ")
			# print (content)
			f.write(qid+"\t"+lineList[1]+"\t"+content+"\t"+dic[lineList[1]])
			sum += 1

	print (qid+" "+str(sum))


# for i in range(2,51):
# 	count_same_number("data_set/clueweb09B-title/qrels.clueweb09b.txt","data_set/clueweb09B-title/document_features_query1_50.txt","data_set/clueweb09B-title/train_data.txt",str(i))

#######################
def get_testFile(inputfile1,inputfile2,outputfile,qid):
	print ("start build testfile!!")
	file1 = open(inputfile1)
	file2 = open(inputfile2)
	outputFile = open(outputfile,"w")
	dic = {}
	flag = 0
	for line in file1.readlines():
		lineList = line.split(' ')
		# print (lineList[0])
		if lineList[0] == qid:
			flag = 1
			dic.update({lineList[1]:lineList[2]})

		# if lineList[0] != qid and flag ==1:
		# 	break
	# print (dic)
	for line in file2.readlines():
		lineList = line.split('\t')
		if lineList[0] == qid and lineList[1] in dic:
			content = str(lineList[2].strip('\n'))
			content = content.strip(" ")
			outputFile.write(qid+"\t"+lineList[1]+"\t"+content+"\t"+dic[lineList[1]]+"\n")
	
	outputFile.close()
	file2.close()
	file1.close()
	print ("get testFile!!")


# for i in range(1,2):
# 	get_testFile("data_set/clueweb09B-title/prels.catB.1-50","data_set/clueweb09B-title/document_features_query1_50.txt","data_set/clueweb09B-title/test_data_1_50.txt",str(i))

# document_features_query1_50.txt

#get word embedding

def read_trainSet(file,file1):
	inputFile = open(file)
	inputfile2 = open(file1)
	outputdic = {}
	qid = 0
	for line in inputFile.readlines():
		# print (line)
		lineList = line.split('\t')
		docID = lineList[2]
		outputdic.update({lineList[0]+"_"+docID:[]})		
		word_count = lineList[3].strip()
		
		word_count = word_count.split(" ")
		# print (word_count)
		
		for word in word_count:
			# word1 = word.split(":")
			# outputdic[docID].update({word1[0]:float(word1[1])})
			outputdic[lineList[0]+"_"+docID].append(word)
	# 	break
	# print (outputdic)
	for line1 in inputfile2.readlines():
		lineList = line1.split('\t')
		docID = lineList[2]
		outputdic.update({lineList[0]+"_"+docID:[]})		
		word_count = lineList[3].strip()
		
		word_count = word_count.split(" ")
		# print (word_count)
		
		for word in word_count:
			# word1 = word.split(":")
			# outputdic[docID].update({word1[0]:float(word1[1])})
			outputdic[lineList[0]+"_"+docID].append(word)

	print ("read_trainSet over!!")
	print ("\n")
	return outputdic

def get_word_dic(doc_wordList):
	# vocab_file = "../data_set/clueweb09B-title/query1_50_vocDic.txt"
	# file = open(vocab_file,"w")
	wid = 0
	word_dic = {}
	for doc in doc_wordList:
		for item in doc_wordList[doc]:
			if item not in word_dic: 
				word_dic.update({item:wid})
				# file.write(str(wid)+"\t"+item+'\n')
				wid += 1
	# file.close()
	print ("\nget_word_dic over!!\n")
	return word_dic

def add_query_dic(word_dic,is_need=False):
	# vocab_file = "../data_set/clueweb09B-title/query1_50_vocDic.txt"
	# file = open(vocab_file,"a+")
	query_file = "../data_set/clueweb09B-title/clueweb.title.krovetz.txt"
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
	# print (query_word)
	max_qid += 1
	for item in query_word:
		if item not in word_dic:
			word_dic.update({item:max_qid})
			# file.write(str(max_qid)+"\t"+item+'\n')
			max_qid += 1
	# file.close()
	file1.close()
	# print ("max_qid")
	# print (max_qid)

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
    print ('embedding_size',embedding_size)
    print ('done')
    print ('words found in wor2vec embedding ',len(vectors.keys()))
    print ("\nget words word2vec over!!\n")
    return vectors

def getSubVectorsFromDict(vectors,vocab,dim = 300):
    # file = open('../embedding/sub_vector_query1_50.txt','w')
    embedding = np.zeros((len(vocab),dim))
    print(len(vocab))
    count = 1
    temp_vec = 0
    for word in vocab:
        # word_em = ""
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

    #     temp_vec += embedding[vocab[word]]
    # temp_vec /= len(vocab)
    # for index,_ in enumerate(embedding):
    # 	embedding[index] -= temp_vec
    	
    print ('word in embedding',count)
    return embedding

# def get_wordDic_Embedding(dim = 50):
# 	build_doc_word_dic = read_trainSet("../data_set/clueweb09B-title/final_train_file.txt","../data_set/clueweb09B-title/final_test_file.txt")
# 	# print (build_doc_word_dic["41_clueweb09-en0006-62-02012"])
# 	# exit()
# 	word_dic = get_word_dic(build_doc_word_dic)
# 	add_query_dic(word_dic,True)
# 	# exit()
# 	# print (len(word_dic.keys()))
# 	sub_vec_file = "../embedding/sub_vector_query1_50.txt"
	
# 	if os.path.exists(sub_vec_file):
# 		sub_embeddings = np.zeros((len(word_dic), dim))
# 		embeddings = {}
# 		f = open(sub_vec_file)
# 		for line in f.readlines():
# 			emb = []
# 			lineList = line.strip().split('\t')
# 			item = lineList[1].strip().split(' ')
# 			for i in item:
# 				emb.append(float(i))
# 			embeddings[lineList[0]] = emb
# 			# break
# 		f.close()
# 		temp_vec = 0
# 		for word in word_dic:
# 			sub_embeddings[word_dic[word]] = embeddings[word]
# 			temp_vec += sub_embeddings[word_dic[word]]
# 		temp_vec /= len(word_dic)
# 		print (len(word_dic))
# 		for index,_ in enumerate(sub_embeddings):
# 			sub_embeddings[index] -= temp_vec
# 	else:
# 		if dim == 50:
# 			fname = "../embedding/aquaint+wiki.txt.gz.ndim=50.bin"
# 			embeddings = KeyedVectors.load_word2vec_format(fname, binary=True)
# 			sub_embeddings = getSubVectors(embeddings,word_dic)
# 		else:
# 			fname = '../embedding/glove.6B/glove.6B.300d.txt'
# 			embeddings = load_text_vec(word_dic,fname,embedding_size = dim)
# 			sub_embeddings = getSubVectorsFromDict(embeddings,word_dic,dim)
# 	print ("get wordDic and Embedding over!!")
# 	return word_dic,sub_embeddings

# word_dic,sub_embedding = get_wordDic_Embedding(300)
# print(sub_embedding[0])
	
def get_wordDic_Embedding(dim = 50):
	build_doc_word_dic = read_trainSet("../pre-process/clueweb09B-title/final_train_file.txt","../pre-process/clueweb09B-title/final_test_file.txt")
	# print (build_doc_word_dic["41_clueweb09-en0006-62-02012"])
	# exit()
	word_dic = get_word_dic(build_doc_word_dic)
	add_query_dic(word_dic,True)
	# exit()
	# print (len(word_dic.keys()))
	
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

# get_wordDic_Embedding(300)

def getSubVectors(vectors,vocab,dim = 50):
 
    for word in vectors:
       	embedding_size = len(vectors[word])
       	break
 
    embedding = np.zeros((len(vocab), embedding_size))
    temp_vec = 0
    for word in vocab:
        if word in vectors.values():
            embedding[vocab[word]]= vectors.word_vec(word)
        else:
            embedding[vocab[word]]= np.random.uniform(-0.25,+0.25,embedding_size)  #.tolist()
        temp_vec += embedding[vocab[word]]
    temp_vec /= len(vocab)
    for index,_ in enumerate(embedding):
    	embedding[index] -= temp_vec
    	break
    print ("getSubVectors over!!")
    return embedding

# embedding = get_Embedding(300)
# print (embedding)
def deal_document_features_file():
	inputFile = open("../data_set/clueweb09B-title/document_features_query1_50.txt")
	outputFile = open("../data_set/clueweb09B-title/document_features_query1_50_Notfidf.txt",'w')
	# f = open(outputFile,"w")
	for line in inputFile.readlines():
		# print (line)
		content = ""
		lineList = line.split('\t')
				
		word_count = lineList[2].split(" ")
		
		for word in word_count:
			word1 = word.split(":")
			# print(word1)
			content += (word1[0].strip()+" ")
		content = content.strip(" ")
		outputFile.write(lineList[0]+"\t"+lineList[1]+"\t"+content+"\n")
		# break
		# print (jilu)
		# if int(lineList[0])!=1:
		# 	break

# deal_document_features_file()

def get_corpus(data_dir):
	# data_dir = "data_set/clueweb09B-title/train_data.txt"
	query_dir = "../data_set/clueweb09B-title/clueweb.title.krovetz.txt"
	outfile = "../data_set/clueweb09B-title/corpus_query1_50.txt"
	f1 = open(data_dir)
	f3 = open(query_dir)
	query_dic = {}
	for query in f3.readlines():
		content = ""
		lineList = query.split("\t")
		for i in range(1,len(lineList)):
			content += lineList[i].strip("\n") + " "
		content = content.strip(" ")
		query_dic.update({lineList[0]:{}})
		query_dic[lineList[0]] = content
	# print (query_dic)
	# exit()
	f2 = open(outfile,'w')
	for line in f1.readlines():
		word = ""
		lineList = line.split("\t")
		word_doc = lineList[2].split(" ")
		for wd in word_doc:
			w = wd.split(":")
			word += w[0]+" "
		word = word.strip(" ")
		f2.write(lineList[0]+"\t"+query_dic[lineList[0]]+"\t"+lineList[1]+"\t"+word+"\t"+lineList[3])
	f2.close()
	f1.close()
# document_features_query1_50
# get_corpus()
#划分文档集
def divide_document_features_Set():
	print ("start divide document set to test and train!!")
	f = open("../pre-process/clueweb09B-title/document_features_query1_150_tfidf_top50.txt")
	out1 = open("../pre-process/clueweb09B-title/train_file.txt",'w')
	out2 = open("../pre-process/clueweb09B-title/test_file.txt",'w')
	qid = []
	for i in range(1,151):
		if i != 100 and i!=95 and i!= 20:
			qid.append(i)
	trainID,testID = train_test_split(qid,test_size=0.2)
	trainID.append(20)
	ind = 0
	for line in f.readlines():
		# if ind > 5:
		# 	break
		words = ""
		tfidfs = []
		w_tfidf = ""
		lineList = line.split("\t")
		if int(lineList[0]) in trainID:
			items = lineList[2].strip()
			items = items.split(" ")
			for item in items:
				# print(item)

				item = item.split(":")
				words += item[0]+" "
				# out3.write(str(lineList[1])+" "+item[0]+"\n")
				# print (lineList[1])
				# print (item)
				# print(item[1])
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

			# ind+=1

		elif int(lineList[0]) in testID:
			items = lineList[2].strip()
			items = items.split(" ")
			for item in items:
				item = item.split(":")
				words += item[0]+" "
				# print (lineList[1])
				# print (item)

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
			# out2.write(line)
			out2.write(lineList[0]+"\t"+lineList[1]+"\t"+words+"\t"+w_tfidf+"\n")

	out2.close()
	out1.close()
	f.close()
	print ("end divide !!")

def get_queryContent_label(data_dir, outfile, test = True):
	if test:
		print ("start build final test file!!")
	else:
		print ("start build final train file!!")
	# data_dir = "data_set/clueweb09B-title/train_data.txt"
	query_dir = "../data_set/clueweb09B-title/clueweb.title.krovetz.txt"
	# label_file = "data_set/clueweb09B-title/prels.catB.1-50"
	label_file = "../qrels_1-150.txt"
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

# get_queryContent_label("data_set/clueweb09B-title/test_file.txt","data_set/clueweb09B-title/final_test_file.txt")

# get_queryContent_label("data_set/clueweb09B-title/train_file.txt","data_set/clueweb09B-title/final_train_file.txt",False)

# def creat_train_test(qid):
# 	divide_document_features_Set()
# 	get_queryContent_label("data_set/clueweb09B-title/test_file.txt","data_set/clueweb09B-title/final_test_file.txt")
# 	get_queryContent_label("data_set/clueweb09B-title/train_file.txt","data_set/clueweb09B-title/final_train_file.txt",False)
# 	# get_corpus()
# 	# data_dir = "data_set/clueweb09B-title/query1.txt"
# 	data_dir = "data_set/clueweb09B-title/corpus_query1_50.txt"
# 	data_train = "data_set/clueweb09B-title/train_query1_50.txt"
# 	data_test = "data_set/clueweb09B-title/test_query1_50.txt"
# 	data_dev = "data_set/clueweb09B-title/dev_query1_50.txt"
# 	datas = []
# 	data_file = os.path.join(data_dir)
# 	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag"],quoting =3).fillna('')
# 	# print(data)
# 	# counter = data.groupby("query_ID").apply(lambda data: sum(data["query_ID"] == int(qid)))
# 	# counter = data.groupby("query_ID").apply(lambda data: sum(data["query_ID"] <= 25))
# 	counter = data.groupby("query_ID").apply(lambda data: sum(data["query_ID"] <= 50))
# 	query_is_qid = counter[counter>0].index
# 	df = data[data["query_ID"].isin(query_is_qid)]
# 	# print (df["flag"])
# 	# x = data[["query_ID","query_content","document","document_content"]]
# 	# y = data["flag"]
# 	# print (questions_have_uncorrect)
# 	train,test = train_test_split(df,test_size=0.3)
# 	# train,test,train_label,test_lable = train_test_split(x,y,test_size=0.2)
# 	train.to_csv(data_train, sep='\t', header=False, index=False)
# 	test.to_csv(data_test, sep='\t', header=False, index=False)
	# print (train)

def load():
	print ("start load train and test datas!!")
	divide_document_features_Set()
	get_queryContent_label("../pre-process/clueweb09B-title/test_file.txt","../pre-process/clueweb09B-title/final_test_file.txt")
	get_queryContent_label("../pre-process/clueweb09B-title/train_file.txt","../pre-process/clueweb09B-title/final_train_file.txt",False)
	data_train = "../pre-process/clueweb09B-title/final_train_file.txt"
	data_test = "../pre-process/clueweb09B-title/final_test_file.txt"

	datas = []
	data_file = os.path.join(data_train)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	
	datas.append(data)
	data_file = os.path.join(data_test)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	datas.append(data)
	print ("end load train and test datas!!")
	return tuple(datas)

# creat_train_test("1")

# load()

def get_query():
	file = "../data_set/clueweb09B-title/clueweb.title.krovetz.txt"
	# file = "clueweb.title.krovetz.txt"
	dic = {}
	f = open(file)
	for line in f.readlines():
		linelist = line.split("\t")
		# print(linelist)
		# print(len(linelist))
		if len(linelist)-1 in dic:
			dic[len(linelist)-1].append(linelist[0])
		else:
			dic[len(linelist)-1] = [linelist[0]]
		
	f.close()
	# print(dic)
	return dic 

# a()

def divide_test_dataset(length):
	# data_test = "../pre-process/clueweb09B-title/final_test_file.txt"
	# out_filePath = "../pre-process/clueweb09B-title/test_file_query_len"+str(length)+".txt"
	data_test = "../pre-process/clueweb09B-title/final_test_file.txt"
	out_filePath = "../pre-process/clueweb09B-title/test_file_query_len"+str(length)+".txt"
	dic = get_query()
	# print(dic[length])
	out_dic = []
	f = open(data_test)
	f1 = open(out_filePath,'w')
	for line in f.readlines():
		linelist = line.split("\t")
		if linelist[0] in dic[length]:
			if linelist[0] not in out_dic:
				out_dic.append(linelist[0])
			f1.write(line)
# 	print(out_dic)

# divide_test_dataset(4)


def divide_test_dataset1(length):
	data_test = "../pre-process/clueweb09B-title/final_test_file.txt"
	out_filePath = "../pre-process/clueweb09B-title/test_file_query_len"+str(length)+".txt"
	# data_test = "final_test_file.txt"
	# out_filePath = "test_file_query_len"+str(length)+".txt"
	dic = get_query()
	# print(dic[length])
	out_dic = []
	f = open(data_test)
	f1 = open(out_filePath,'w')
	for line in f.readlines():
		linelist = line.split("\t")
		# if linelist[0] not in dic[1]:
		if linelist[0] not in out_dic:
			out_dic.append(linelist[0])
		f1.write(line)
# 	print(out_dic)
# divide_test_dataset1(2)

def load_test_apply(length):
	print ("start load test datas!!")
	# divide_document_features_Set()
	# get_queryContent_label("../pre-process/clueweb09B-title/test_file.txt","../pre-process/clueweb09B-title/final_test_file.txt")
	# get_queryContent_label("../pre-process/clueweb09B-title/train_file.txt","../pre-process/clueweb09B-title/final_train_file.txt",False)
	# data_train = "../pre-process/clueweb09B-title/final_train_file.txt"
	# data_test = "../pre-process/clueweb09B-title/final_test_file.txt"
	# out_filePath = "../pre-process/clueweb09B-title/test_file_query_len1.txt" 

	# divide_test_dataset(length)
	divide_test_dataset1(length)
	data_test = "../pre-process/clueweb09B-title/test_file_query_len"+str(length)+".txt"
	datas = []
	# data_file = os.path.join(data_train)
	# data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	
	# datas.append(data)
	data_file = os.path.join(data_test)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","query_content","document","document_content","flag","tfidf"],quoting =3).fillna('')
	datas.append(data)
	print ("end load test datas!!")
	# return tuple(datas)
	return data

# test = load_test_apply(1)
# print(test)

@log_time_delta
def get_overlap_dict(df,alphabet,q_len = 40,d_len = 40):
    d = dict()
    # print(df['query_content'])
    # print(df['query_content'].unique())
    # exit()
    for query in df['query_content'].unique():
        group = df[df['query_content'] == query]
        document_content = group['document_content']
        for doc in document_content:
            q_overlap,d_overlap = overlap_index(query,doc,q_len,d_len)
            d[(query,doc)] = (q_overlap,d_overlap)
    return d
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

def cut(sentence):
    tokens = sentence.lower().split()
    return tokens

def position_index(sentence,length):
    index = np.zeros(length)

    raw_len = len(cut(sentence))
    index[:min(raw_len,length)] = range(1,min(raw_len + 1,length + 1))
    # print index
    return index

@log_time_delta
def batch_gen_with_single(df,alphabet,batch_size = 10,q_len = 33,d_len = 40,overlap_dict = None):
    pairs=[]
    input_num = 6
    for index,row in df.iterrows():
        query = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
        document = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
        if overlap_dict == None:
            q_overlap,d_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
        else:
            q_overlap,d_overlap = overlap_dict[(row["query_content"],row["document_content"])]
        q_position = position_index(row['query_content'],q_len)
        d_position = position_index(row['document_content'],d_len)
        pairs.append((query,document,q_overlap,d_overlap,q_position,d_position))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs)*1.0 / batch_size)
    # pairs = sklearn.utils.shuffle(pairs,random_state =132)
    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]

        yield [[pair[j] for pair in batch]  for j in range(input_num)]
    batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield [[pair[i] for pair in batch]  for i in range(input_num)]

@log_time_delta
def batch_gen_with_test(df,alphabet,batch_size = 10,q_len = 33,d_len = 40,overlap_dict = None):
	pairs=[]
	# input_num = 2
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
	# batch= pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
	# yield [[pair[i] for pair in batch]  for i in range(input_num)]
	
	if (len(pairs)*1.0 % batch_size) == 0:
		batch =[pairs[n_batches*batch_size-1]] * (batch_size-len(pairs)+n_batches*batch_size)
		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]
	else:
		batch = pairs[n_batches*batch_size:]+[pairs[n_batches*batch_size]] * (batch_size-len(pairs)+n_batches*batch_size)
		yield [np.array([pair[j] for pair in batch]) for j in range(input_num)]



def encode_to_split(sentence,alphabet,max_sentence = 40):
    indices = []    
    tokens = cut(sentence)
    for word in tokens:
        indices.append(alphabet[word])
    while(len(indices) < max_sentence):
        indices += indices[:(max_sentence - len(indices))]
    # results=indices+[alphabet["END"]]*(max_sentence-len(indices))
    return indices[:max_sentence]

def get_tfidf_value(tfidf_value,max_qlen = 4,max_dlen = 50):
	return_list =[]
	zeros_list = []
	sum_value = 0
	tokens = cut(tfidf_value)
	# print(tokens)
	for i in tokens:
		return_list.append(float(i))
		# sum_value += float(i)
	for i in range(max_dlen - len(return_list)):
		zeros_list.append(0)

	while(len(return_list) < max_dlen):
		return_list += zeros_list[:max_dlen - len(return_list)]

	sum_return_list = sum(return_list)

	for index,item in enumerate(return_list):
		return_list[index] = item/(sum_return_list*max_qlen)	

	return np.array(return_list[:max_dlen])

def transform(flag):
    # if flag == 1:
    #     return [0,1,0]
    # elif flag == 0:
    #     return [1,0,0]
    # elif flag == 2:
    # 	return [0,0,1]

    # if flag == 1:
    #     return np.array([1], dtype = float)
    # elif flag == 0:
    #     return np.array([0], dtype = float)
    # elif flag == 2:
    # 	return np.array([2], dtype = float)
    return np.array([float(flag)], dtype = float)
    # if flag == 1:
    #     return np.array([1.0], dtype = float)
    # elif flag == 0:
    #     return np.array([0.0], dtype = float)
    # elif flag == 2:
    # 	return np.array([2.0], dtype = float)
    # if flag == 1:
    #     return 1.0
    # elif flag == 0:
    #     return 0.0
    # elif flag == 2:
    # 	return 2.0

@log_time_delta
def batch_gen_with_point_wise(df,alphabet, batch_size = 10,overlap_dict = None,q_len = 33,d_len = 40):
    #inputq inputa intput_y overlap
    input_num = 7
    pairs = []
    for index,row in df.iterrows():
        question = encode_to_split(row["query_content"],alphabet,max_sentence = q_len)
        answer = encode_to_split(row["document_content"],alphabet,max_sentence = d_len)
   
        q_overlap,a_overlap = overlap_index(row["query_content"],row["document_content"],q_len,d_len)
        q_position = position_index(row['query_content'],q_len)
        a_position = position_index(row['document_content'],d_len)
        label = transform(row["flag"])
        pairs.append((question,answer,label,q_overlap,a_overlap,q_position,a_position))
    # n_batches= int(math.ceil(df["flag"].sum()*1.0/batch_size))
    n_batches = int(len(pairs)*1.0 / batch_size)
    pairs = sklearn.utils.shuffle(pairs,random_state = 121)

    for i in range(0,n_batches):
        batch = pairs[i*batch_size:(i+1) * batch_size]
        yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]
    print ("****************************************")
    print ("n_batches")
    print (n_batches)
    print ("list index:")
    print (n_batches*batch_size)
    batch = pairs[n_batches*batch_size:] + [pairs[n_batches*batch_size]] * (batch_size- len(pairs)+n_batches*batch_size  )
    yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]

@log_time_delta
def batch_gen_with_list_wise(df,alphabet, batch_size = 10,q_len = 4, d_len = 50,overlap_dict = None):
    #inputq inputa intput_y overlap
    # input_num = 7
    # input_num = 3
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
        #打乱
        shuffle(pairs)       
        n_batches = int(len(pairs)*1.0 / batch_size)
        # print (n_batches)
        for i in range(0,n_batches):
            batch = pairs[i*batch_size:(i+1) * batch_size]
            yield [np.array([pair[i] for pair in batch])  for i in range(input_num)]
    #     if len(pairs[n_batches*batch_size:])>0:
    #         leave_data = pairs[n_batches*batch_size:]
    #         for pair in leave_data:
    #             no_useList.append(pair)

    # shuffle(no_useList)
    # print (len(no_useList))
    # batch_end = no_useList[:batch_size]
    # print (len(batch_end))
    # yield [np.array([pair[i] for pair in batch_end])  for i in range(input_num)]
    
    
# train,test = load()
# alphabet,embeddings = get_wordDic_Embedding(300)
# datas = batch_gen_with_list_wise(train, alphabet, 20, q_len = 4, d_len = 50)
# i = 1
# for data in datas:
# 	print (data)
# 	# if i>4:
# 	# 	break
# 	i+=1



# d = get_overlap_dict(train,word_dic,3,21)
# print (len(train))
# datas = batch_gen_with_point_wise(train,word_dic,10,overlap_dict = d,
# 					q_len = 3,a_len = 21)
# for data in datas:
	# print (data[0])
	# print (data[6])
# for data in batch_gen_with_single(train,word_dic,10,3,21,overlap_dict = d):
# 	print (len(data[0]))
# 	print ("***************")
# 	print (data[0])
# 	print (len(data[1]))
# 	# print (data[2])
# 	# for i in range(0,6):
# 	# 	print (data[i])


# train,test = load()
# alphabet,embeddings = get_wordDic_Embedding(300)
# # print (train)
# # print (test)
# i=0
# for data in batch_gen_with_test(test,alphabet,30,4,21):
# 	# data["clueweb09-en0000-60-10466"]
# 	i+=1