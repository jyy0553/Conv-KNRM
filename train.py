#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
# from pre_dateset import batch_gen_with_point_wise,load,prepare,batch_gen_with_single
# from pre_dateset import load,get_wordDic_Embedding,batch_gen_with_test,batch_gen_with_list_wise,get_overlap_dict
from units import load,get_wordDic_Embedding,batch_gen_with_test,batch_gen_with_list_wise,get_overlap_dict,batch_gen_with_pair_wise
import operator
from model import IR_quantum
import random
import evaluation as evaluation_test
# import evaluation_test
# import cPickle as pickle
import pickle
from sklearn.model_selection import train_test_split
import configure
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tensorboard_log_dir =  "tensorboard_logs/"
now = int(time.time()) 
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps

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

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'log/'+ timeDay+"/"+FLAGS.data+"/"+FLAGS.file_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'

# pickle.dump(FLAGS.__flags,open(para_file,'wb'))

def kernal_mus(n_kernels, use_exact=True):
	"""
	get the mu for each guassian kernel. Mu is the middle of each bin
	:param n_kernels: number of kernels (including exact match). first one is exact match
	:return: l_mu, a list of mu.
	"""
	if use_exact:
		l_mu = [1]
	else:
		l_mu = [2]
	if n_kernels == 1:
		return l_mu

	bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
	l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
	for i in range(1, n_kernels - 1):
		l_mu.append(l_mu[i] - bin_size)
	return l_mu

def kernel_sigmas(n_kernels, lamb, use_exact=True):
	"""
	get sigmas for each guassian kernel.
	:param n_kernels: number of kernels (including exactmath.)
	:param lamb:
	:param use_exact:
	:return: l_sigma, a list of simga
	"""
	bin_size = 2.0 / (n_kernels - 1)
	l_sigma = [0.00001]  # for exact match. small variance -> exact match
	if n_kernels == 1:
		 return l_sigma
	l_sigma += [bin_size * lamb] * (n_kernels - 1)
	return l_sigma

@log_time_delta
def predict(sess,cnn,test,alphabet, batch_size,q_len,d_len):
	scores = []
	d = get_overlap_dict(test,alphabet,q_len,d_len)
	# for data in batch_gen_with_single(test,alphabet,batch_size,q_len,d_len,overlap_dict = d):
	for data in batch_gen_with_test(test,alphabet,batch_size,q_len,d_len,overlap_dict = d):
		feed_dict = {
			cnn.query:data[0],
			cnn.pos_doc:data[1],			
			cnn.dropout_keep_prob:1.0,
			cnn.input_mu:kernal_mus(FLAGS.kener_num),
			cnn.input_sigma:kernel_sigmas(FLAGS.kener_num,0.5)
		}
		score = sess.run(cnn.pos_scores,feed_dict)
		# print ("score_len")
		# print (len(score))
		scores.extend(score)
	return np.array(scores[:len(test)])

@log_time_delta
def test_point_wise():
	# creat_train_test("2")
	train,test = load(FLAGS.data,FLAGS.file_name)
	# train,test,dev = load(FLAGS.data,filter = FLAGS.clean)
	# print ()
	# q_max_sent_length = 4
	q_max_sent_length = FLAGS.max_len_query
	# d_max_sent_length = 21
	d_max_sent_length = FLAGS.max_len_document

	# alphabet,embeddings = prepare([train,test,dev],dim = FLAGS.embedding_dim,is_embedding_needed = True,fresh = True)
	# alphabet,embeddings = get_wordDic_Embedding(300)
	alphabet,embeddings = get_wordDic_Embedding(FLAGS.data, FLAGS.file_name, 50)
	print ("alphabet",len(alphabet))
	# exit()
	with tf.Graph().as_default():
		with tf.device("/gpu:0"):
			session_conf = tf.ConfigProto()
			session_conf.allow_soft_placement = FLAGS.allow_soft_placement
			session_conf.log_device_placement = FLAGS.log_device_placement
			session_conf.gpu_options.allow_growth = True
		sess = tf.Session(config = session_conf)
		with sess.as_default(),open(precision,"w") as log:
			log.write(str(FLAGS.__flags)+'\n')
			cnn = IR_quantum(
				max_input_query = q_max_sent_length,
				max_input_docu = d_max_sent_length,
				vocab_size = len(alphabet),
				embedding_size = FLAGS.embedding_dim,
				batch_size = FLAGS.batch_size,
				embeddings = embeddings,
				filter_sizes = list(map(int,FLAGS.filter_sizes.split(","))),
				num_filters = FLAGS.num_filters,
				l2_reg_lambda = FLAGS.l2_reg_lambda,
				trainable = FLAGS.trainable,
				overlap_needed = FLAGS.overlap_needed,
				pooling = FLAGS.pooling,
				extend_feature_dim = FLAGS.extend_feature_dim
				)
			cnn.build_graph()
			
			global_step = tf.Variable(0,name = 'global_step',trainable = False)
			learning_rate = FLAGS.learning_rate
			# learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,100,0.96)
			# optimizer = tf.train.GradientDescentOptimizer(learning_rate,global_step = global_step)


			# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

			# optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=1e-08)
			# grads_and_vars = optimizer.compute_gradients(cnn.loss)
			# train_op = optimizer.apply_gradients(grads_and_vars,global_step = global_step)

			# optimizer
			train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08).minimize(cnn.loss)

			saver = tf.train.Saver(tf.global_variables(),max_to_keep = 4)
			
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(tensorboard_log_dir+"/train",sess.graph)
			
			sess.run(tf.global_variables_initializer())

			map_max = 0.020
			# loss_max = 0.3
			for i in range(FLAGS.num_epochs):
				print ("\nepoch "+str(i)+"\n")
				d = get_overlap_dict(train,alphabet,q_len = q_max_sent_length,d_len = d_max_sent_length)

				# datas = batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap_dict = d,
				# 	q_len = q_max_sent_length,d_len = d_max_sent_length)
				# datas = batch_gen_with_list_wise(train,alphabet,FLAGS.batch_size,q_len = q_max_sent_length,d_len = d_max_sent_length,overlap_dict = d)
				datas = batch_gen_with_pair_wise(train,alphabet,FLAGS.batch_size,q_len = q_max_sent_length,d_len = d_max_sent_length,overlap_dict = d)
				

				# if i <2:
				# 	continue
				j = 1
				for data in datas:			

					feed_dict = {
						cnn.query:data[0][0],
						cnn.pos_doc:data[0][1],
						cnn.neg_doc:data[1][1],
						cnn.input_label:data[0][2],						
						cnn.dropout_keep_prob:0.5,
						cnn.input_mu:kernal_mus(FLAGS.kener_num),
						cnn.input_sigma:kernel_sigmas(FLAGS.kener_num,0.5)
					}
					_, pos_scores, neg_scores, loss = sess.run([train_op, cnn.pos_scores, cnn.neg_scores, cnn.loss], feed_dict)

					# print ("pos_scores : {}".format(pos_scores))
					# print ("neg_scores : {}".format(neg_scores))					
					print ("{} loss: {}".format(j,loss))

					# exit()

					j+=1
					# # print ("para")
					# # print (para)
					# print ("score"+str(scores))
					time_str = datetime.datetime.now().isoformat()

				predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,d_max_sent_length)
				# print ("train predict")
				# print (predicted[:,-1])
				map_NDCG0_NDCG1_ERR_p_train = evaluation_test.evaluationBypandas(train,predicted[:,-1])
				# precision_train = evaluation.precision(train,predicted[:,-1])
				# predicted = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,d_max_sent_length)
				# map_mrr_dev = evaluation.evaluationBypandas(dev,predicted[:,-1])
				# precision_dev = evaluation.precision(dev,predicted[:,-1])
				predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,d_max_sent_length)
				
				# print ("test predict")
				# print (predicted_test[:,-1])
				
				map_NDCG0_NDCG1_ERR_p_test = evaluation_test.evaluationBypandas(test,predicted_test[:,-1])
				# precision_test = evaluation.precision(test,predicted_test[:,-1])

				if map_NDCG0_NDCG1_ERR_p_test[0] > map_max:
					map_max = map_NDCG0_NDCG1_ERR_p_test[0]
					timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
					folder = 'runs/'+FLAGS.data+"/"+FLAGS.file_name+"/"+ timeDay
					out_dir = folder +'/'+timeStamp+'_'+str(map_NDCG0_NDCG1_ERR_p_test[0])
					if not os.path.exists(folder):
						os.makedirs(folder)
					save_path = saver.save(sess, out_dir)
					print ("Model saved in file: ", save_path)

				print ("{}:train epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_train))
				# print('precision_train',precision_train)
				# print ("{}:dev epoch:map mrr {}".format(i,map_mrr_dev))
				# print('precision_dev',precision_dev)
				# f = open()
				print ("{}:test epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_test))
				# file = "result/listwise_"+timeDay+"_learnrate_"+str(FLAGS.learning_rate)+".txt"
				# f = open(file,"a")
				# f.write("{}:train epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_train))
				# f.write("{}:test epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_test))
				# f.write("\n")
				# f.close()
				# print('precision_test',precision_test)
				# line = " {}:epoch: map_test{},precision_test: {}".format(i,map_mrr_test,precision_test)
				
				line1 = " {}:epoch: map_train{}".format(i,map_NDCG0_NDCG1_ERR_p_train)
				log.write(line1+"\n")
				line = " {}:epoch: map_test{}".format(i,map_NDCG0_NDCG1_ERR_p_test)
				log.write(line+"\n")
				log.write("\n")
				log.flush()
			log.close()

if __name__ == "__main__":
	if FLAGS.loss == "point_wise":
		test_point_wise()