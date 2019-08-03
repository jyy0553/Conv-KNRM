# coding:utf-8

import tensorflow as tf

import numpy as np 


t = np.array([1, 2, 3,4])

a1 = np.random.choice(t,3,False)
a2= np.random.choice(t,3,False)
b = np.hstack((a1,a2))
print (a1)
print (a2)
print (b)
# a1 = np.random.choice(a=5, size=3, replace=False, p=None)
# print(a1)


# t = tf.constant([[[1, 1, 1], [2, 2, 2], [7, 7, 7]],
#                  [[3, 3, 3], [4, 4, 4], [8, 8, 8]],
#                  [[5, 5, 5], [6, 6, 6], [9, 9, 9]]])

# z1 = tf.strided_slice(t, [0], [3], [1])
# z2 = tf.strided_slice(t, [1, 0], [-1, 2], [1, 1])
# z3 = tf.strided_slice(t, [1, 0, 1], [-1, 2, 3], [1, 1, 1])


# t1 = tf.constant([[1.,1.],[3.,3.],[2.,2.]])
# w = tf.constant([[1.,1.],[1.,1.]])
# r1 = tf.matmul(t1,w)
# t2 = tf.transpose(t1, perm=[1,0])
# r2 = tf.matmul(r1,t2)
# diag = tf.nn.softmax(tf.diag_part(r2))
# sum_ = tf.reduce_sum(diag)
# index = tf.arg_max(diag,0)

# # log2 = tf.log(2.,2)
# batch_identity = tf.eye(2, batch_shape=[3])


# with tf.Session() as sess:
#     # print(sess.run(z1))
#     # print (t.get_shape())
#     # print (z1.get_shape())
#     # print(sess.run(z2))

#     # print(sess.run(z3))

#     print (t1.get_shape())
#     print (sess.run(r1))
#     print (r1.get_shape())
#     print (sess.run(r2))
#     print (sess.run(diag))
#     # print (sess.run(sum_))
#     print (sess.run(index))
#     # print (sess.run(log2))
#     print (sess.run(batch_identity))