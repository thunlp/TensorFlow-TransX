#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import ctypes

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")
test_lib = ll("./test.so")

class Config(object):

	def __init__(self):
		lib.setInPath("./data/FB15K/")
		test_lib.setInPath("./data/FB15K/")
		lib.setBernFlag(0)
		self.learning_rate = 0.0001
		self.testFlag = False
		self.loadFromData = False
		self.L1_flag = True
		self.hidden_sizeE = 100
		self.hidden_sizeR = 100
		self.nbatches = 100
		self.entity = 0
		self.relation = 0
		self.trainTimes = 1000
		self.margin = 1.0
		'''
		In the original paper, TransR is trained with the pre-trained embeddings as parameter initialization.
		If you do not want to train with the pre-trained embeddings, you can use the following code instead of the default version.
		You need to use np.savetxt() to store the pre-trained embeddings into the corresponding file and input the file's name.
		e.g.
			self.ent_init = "ent_embeddings.txt"
			self.rel_init = "rel_embeddings.txt"
		where entity and relation embeddings are stored into the "ent_embeddings.txt" and "rel_embeddings.txt" respectively.
		'''
		self.rel_init = None
		self.ent_init = None
		
class TransRModel(object):

	def __init__(self, config, ent_init = None, rel_init = None):

		entity_total = config.entity
		relation_total = config.relation
		batch_size = config.batch_size
		sizeE = config.hidden_sizeE
		sizeR = config.hidden_sizeR
		margin = config.margin

		with tf.name_scope("read_inputs"):
			self.pos_h = tf.placeholder(tf.int32, [batch_size])
			self.pos_t = tf.placeholder(tf.int32, [batch_size])
			self.pos_r = tf.placeholder(tf.int32, [batch_size])
			self.neg_h = tf.placeholder(tf.int32, [batch_size])
			self.neg_t = tf.placeholder(tf.int32, [batch_size])
			self.neg_r = tf.placeholder(tf.int32, [batch_size])

		with tf.name_scope("embedding"):
			if ent_init != None:
				self.ent_embeddings = tf.Variable(np.loadtxt(ent_init), name = "ent_embedding", dtype = np.float32)
			else:
				self.ent_embeddings = tf.get_variable(name = "ent_embedding", shape = [entity_total, sizeE], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			if rel_init != None:
				self.rel_embeddings = tf.Variable(np.loadtxt(rel_init), name = "rel_embedding", dtype = np.float32)
			else:
				self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [relation_total, sizeR], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			
			rel_matrix = np.zeros([relation_total, sizeR * sizeE], dtype = np.float32)
			for i in range(relation_total):
				for j in range(sizeR):
					for k in range(sizeE):
						if j == k:
							rel_matrix[i][j * sizeE + k] = 1.0
			self.rel_matrix = tf.Variable(rel_matrix, name = "rel_matrix")

		with tf.name_scope('lookup_embeddings'):
			pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, sizeE, 1])
			pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, sizeE, 1])
			pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, sizeR])
			neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, sizeE, 1])
			neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, sizeE, 1])
			neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, sizeR])			
			pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.pos_r), [-1, sizeR, sizeE])
			neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r), [-1, sizeR, sizeE])

			pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e), [-1, sizeR]), 1)
			pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e), [-1, sizeR]), 1)
			neg_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_h_e), [-1, sizeR]), 1)
			neg_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_t_e), [-1, sizeR]), 1)

		if config.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
			self.predict = pos
		else:
			pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
			neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
			self.predict = pos

		with tf.name_scope("output"):
			self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

def main(_):
	config = Config()
	if (config.testFlag):
		test_lib.init()
		config.relation = test_lib.getRelationTotal()
		config.entity = test_lib.getEntityTotal()
		config.batch = test_lib.getEntityTotal()
		config.batch_size = config.batch
	else:
		lib.init()
		config.relation = lib.getRelationTotal()
		config.entity = lib.getEntityTotal()
		config.batch_size = lib.getTripleTotal() // config.nbatches

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				trainModel = TransRModel(config = config, ent_init = config.ent_init, rel_init = config.rel_init)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(config.learning_rate)
			grads_and_vars = optimizer.compute_gradients(trainModel.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())
			if (config.loadFromData):
				saver.restore(sess, 'model.vec')

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

			def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
				}
				step, predict = sess.run(
					[global_step, trainModel.predict], feed_dict)
				return predict

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

			lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
			test_lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
			test_lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
			test_lib.testHead.argtypes = [ctypes.c_void_p]
			test_lib.testTail.argtypes = [ctypes.c_void_p]

			if not config.testFlag:
				for times in range(config.trainTimes):
					res = 0.0
					for batch in range(config.nbatches):
						lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
						res += train_step(ph, pt, pr, nh, nt, nr)
						current_step = tf.train.global_step(sess, global_step)
					print times
					print res
				saver.save(sess, 'model.vec')
			else:
				total = test_lib.getTestTotal()
				for times in range(total):
					test_lib.getHeadBatch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.testHead(res.__array_interface__['data'][0])

					test_lib.getTailBatch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.testTail(res.__array_interface__['data'][0])
					print times
					if (times % 50 == 0):
						test_lib.test()
				test_lib.test()

if __name__ == "__main__":
	tf.app.run()

