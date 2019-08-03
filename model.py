#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf 
import numpy as np 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
rng = np.random.RandomState(23455)


class IR_quantum(object):
	def __init__(
		self, max_input_query,max_input_docu, vocab_size, embedding_size ,batch_size,
		embeddings,filter_sizes,num_filters,l2_reg_lambda = 0.0,trainable = True,
		pooling = 'max',overlap_needed = True,extend_feature_dim = 10):

		# self.dropout_keep_prob = dropout_keep_prob
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.trainable = trainable
		self.filter_sizes = filter_sizes
		self.pooling = pooling
		self.total_embedding_dim = embedding_size
		self.batch_size = batch_size
		self.l2_reg_lambda = l2_reg_lambda
		self.para = []
		self.max_input_query = max_input_query
		self.max_input_docu = max_input_docu
		self.hidden_num = 128
		self.rng = 23455
		self.overlap_need = overlap_needed
		# if self.overlap_need:
		# 	self.total_embedding_dim = embedding_size + extend_feature_dim
		# else:
		# 	self.total_embedding_dim = embedding_size
		self.extend_feature_dim = extend_feature_dim
		self.conv1_kernel_num = 32
		# self.conv2_kernel_num = 32
		self.n_bins = 11
		self.stdv = 0.5
		self.num_filters = 128
		kernel_sizes = [1,2,3]
		print (self.max_input_query)
		print (self.max_input_docu)


	def weight_variable(self,shape):
		tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
		initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
		return tf.Variable(initial)

	
	def creat_placeholder(self):
		self.query = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "input_query")
		self.pos_doc = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "pos_document")		
		self.neg_doc = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "neg_document")

		self.input_label = tf.placeholder(tf.float32,[self.batch_size,1],name = "input_label")

		# self.q_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "q_overlap")
		# self.d_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "d_overlap")
		# self.tfidf_value = tf.placeholder(tf.float32,[self.batch_size,self.max_input_docu],name = 'tfidf_value')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name ="dropout_keep_prob")

		self.input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
		self.input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')

		# self.neg_scores = tf.placeholder(tf.float32, shape = [self.batch_size,1], name = "neg_scores")
		# self.pos_scores = tf.placeholder(tf.float32, shape = [self.batch_size,1], name = "pos_scores")


		self.mu = tf.reshape(self.input_mu, shape=[1, 1, self.n_bins])
		self.sigma = tf.reshape(self.input_sigma, shape=[1, 1, self.n_bins])
	
		# self.W1 = self.weight_variable([self.n_bins,1])
		self.b1 = tf.Variable(tf.zeros([1]))
		length = pow(len(self.kernel_sizes),2)
		self.W1 = self.weight_variable([self.n_bins * length,1])
		self.q_weights = tf.Variable(tf.float32, shape=[self.batch_size, self.max_input_docu], name = 'idf')

	def load_embeddings(self):


		self.words_embeddings = tf.Variable(np.array(self.embeddings),name = "word_emb",dtype = "float32",trainable = False)
		
		self.pos_doc_emb = tf.nn.embedding_lookup(self.words_embeddings, self.pos_doc, name="pos_doc_emb")
		self.neg_doc_emb = tf.nn.embedding_lookup(self.words_embeddings, self.neg_doc, name="neg_doc_emb")
		self.query_emb = tf.nn.embedding_lookup(self.words_embeddings, self.query, name="query_emb")
		

	def get_loss(self):
		# self.pos_scores = self.model(self.normalized_q_embed, self.pos_tmp)
		# self.neg_scores = self.model(self.normalized_q_embed, self.neg_tmp)

		self.pos_scores = self.conv_model(self.query_emb,self.pos_doc_emb)
		self.neg_scores = self.conv_model(self.query_emb, self.neg_doc_emb)
		self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_scores + self.neg_scores))


	def model(self, query_emb, doc_emb):
		# similarity matrix [n_batch, qlen, dlen]
		self.sim = tf.matmul(query_emb, doc_emb, name='similarity_matrix')

		# print ("sim shape : {}".format(self.sim.get_shape()))
		# exit()
		# compute gaussian kernel
		rs_sim = tf.reshape(self.sim, [self.batch_size, self.max_input_query, self.max_input_docu, 1])

		tmp_model = tf.exp(-tf.square(tf.subtract(rs_sim, self.mu)) / (tf.multiply(tf.square(self.sigma), 2)))

		feats = []

		kde = tf.reduce_sum(tmp_model,[2])
		kde = tf.log(tf.maximum(kde,1e-10))*0.01

		# aggregated_kde = tf.reduce_sum(kde * q_weights, [1])

		aggregated_kde = tf.reduce_sum(kde, [1])

		feats.append(aggregated_kde)
		feats_tmp = tf.concat(feats,1)

		feats_flat = tf.reshape(feats_tmp,[-1,self.n_bins])

		lo = tf.matmul(feats_flat, self.W1)+self.b1 

		scores = tf.tanh(lo)

		return scores
		# print ("scores shape : {}".format(self.scores.get_shape()))
		# self.scores = self.logits
		# exit()


	# def create_loss(self):
	# 	self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_scores + self.neg_scores))
	def conv_model(self, query_emb, doc_emb):
		with tf.variable_scope("cnn",reuse = tf.AUTO_REUSE):
			q_convs = []
			d_convs = []
			for size in self.kernel_sizes:
				conv_q = tf.layers.conv1d(query_emb,self.num_filters,size,padding = "same",activation = tf.nn.relu)
				conv_d = tf.layers.conv1d(doc_emb,self.num_filters,size,padding = "same", activation = tf.nn.relu)
				q_convs.append(conv_q)
				d_convs.append(conv_d)

			simis = []
			normalized_q_convs = []
			for q_conv in q_convs:
				norm_q_conv = tf.sqrt(tf.reduce_sum(tf.square(q_conv),1,keep_dim = True))
				normalized_q_conv = q_conv/norm_q_conv
				normalized_q_convs.append(normalized_q_conv)
			normalized_d_conv_ts = []
			for d_conv in d_convs:
				norm_d_conv = tf.sqrt(tf.reduce_sum(tf.square(d_conv),2,keep_dim = True))
				normalized_d_donv = d_conv /norm_d_conv
				normalized_d_conv_t = tf.transpose(normalized_d_donv, perm=[0, 2, 1])
				normalized_d_conv_ts.append(normalized_d_conv_t)

			for normalized_q_conv in normalized_q_convs:
				for normalized_d_conv_t in normalized_d_conv_ts:
					simi = tf.matmul(normalized_q_conv, normalized_d_conv_t)
					simis.append(simi)
		with tf.variable_scope("kernel", reuse = tf.AUTO_REUSE):
			feats = []
			for sim in simis:
				rs_sim = tf.reshape(sim,[self.batch_size,self.max_input_query,self.max_input_docu,1])
				tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mu))/(tf.multiply(tf.square(self.sigma),2)))
				kde = tf.reduce_sum(tmp,[2])
				kde = tf.log1p(kde)
				aggregated_kde = tf.reduce_sum(kde * self.q_weights,[1])
				feats.append(aggregated_kde)
		with tf.variable_scope('fc',reuse = tf.AUTO_REUSE):
			feats_tmp = tf.concat(feats,1)
			lo = tf.matmul(feats_tmp,self.W1) + self.b1

			scores = tf.tanh(lo)
		return scores

	def build_graph(self):
		self.creat_placeholder()
		self.load_embeddings()
		# self.model()
		# self.create_loss()
		self.get_loss()
		print ("end build graph")

