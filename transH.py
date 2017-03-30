#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")

class Config(object):

	def __init__(self):
		self.L1_flag = True
		self.hidden_size = 100
		self.nbatches = 100
		self.entity = 0
		self.relation = 0
		self.trainTimes = 3000
		self.margin = 1.0

class TransHModel(object):

	def calc(self, e, n):
		norm = tf.nn.l2_normalize(n, 1)
		return e - tf.reduce_sum(e * norm, 1, keep_dims = True) * norm

	def __init__(self, config):

		entity_total = config.entity
		relation_total = config.relation
		batch_size = config.batch_size
		size = config.hidden_size
		margin = config.margin

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])

		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.normal_vector = tf.get_variable(name = "normal_vector", shape = [relation_total, size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			
			pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
			
			neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
			
			pos_norm = tf.nn.embedding_lookup(self.normal_vector, self.pos_r)
			neg_norm = tf.nn.embedding_lookup(self.normal_vector, self.neg_r)

			pos_h_e = self.calc(pos_h_e, pos_norm)
			pos_t_e = self.calc(pos_t_e, pos_norm)
			neg_h_e = self.calc(neg_h_e, neg_norm)
			neg_t_e = self.calc(neg_t_e, neg_norm)

		if config.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
		else:
			pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
			neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)	

		with tf.name_scope("output"):
			self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

def main(_):
	lib.init()
	config = Config()
	config.relation = lib.getRelationTotal()
	config.entity = lib.getEntityTotal()
	config.batch_size = lib.getTripleTotal() / config.nbatches

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				trainModel = TransHModel(config = config)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.GradientDescentOptimizer(0.001)
			grads_and_vars = optimizer.compute_gradients(trainModel.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())

			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
					trainModel.neg_h: neg_h_batch,
					trainModel.neg_t: neg_t_batch,
					trainModel.neg_r: neg_r_batch
				}
				_, step, loss = sess.run(
					[train_op, global_step, trainModel.loss], feed_dict)
	 			return loss

	 		ph = np.zeros(config.batch_size, dtype = np.int32)
	 		pt = np.zeros(config.batch_size, dtype = np.int32)
	 		pr = np.zeros(config.batch_size, dtype = np.int32)
	 		nh = np.zeros(config.batch_size, dtype = np.int32)
	 		nt = np.zeros(config.batch_size, dtype = np.int32)
	 		nr = np.zeros(config.batch_size, dtype = np.int32)

	 		ph_addr = ph.__array_interface__['data'][0]
	 		pt_addr = pt.__array_interface__['data'][0]
	 		pr_addr = pr.__array_interface__['data'][0]
	 		nh_addr = nh.__array_interface__['data'][0]
	 		nt_addr = nt.__array_interface__['data'][0]
	 		nr_addr = nr.__array_interface__['data'][0]
	
	 		for times in range(config.trainTimes):
	 			res = 0.0
				for batch in range(config.nbatches):
					lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
					res += train_step(ph, pt, pr, nh, nt, nr)
					current_step = tf.train.global_step(sess, global_step)
				print times
				print res
			saver.save(sess, 'model.vec')

if __name__ == "__main__":
	tf.app.run()

