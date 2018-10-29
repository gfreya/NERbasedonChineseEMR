# -*- coding: utf-8 -*
from tensorflow.contrib.rnn import DropoutWrapper
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from utils import *

BATCH_SIZE = config.FLAGS.batch_size # 每个周期内训练批处理的样本数
unit_num = embeddings_size  # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = max_sequence  # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = config.FLAGS.dropout # dropout层神经元被丢弃的概率
EPOCH = config.FLAGS.epoch # 训练周期数
TAGS_NUM = get_class_size() #获取命名实体识别类别总数


class NER_net:
	def __init__(self, scope_name, iterator, embedding, batch_size):
		'''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size: 每个周期内训练批处理的样本数
        '''
		self.batch_size = batch_size
		self.embedding = embedding
		self.iterator = iterator
		with tf.variable_scope(scope_name) as scope:
			self._build_net()

	def _build_net(self):
		self.global_step = tf.Variable(0, trainable=False)
		source = self.iterator.source
		tgt = self.iterator.target_input
		# 得到当前batch的长度（如果长度不足的会被padding填充）
		max_sequence_in_batch = self.iterator.source_sequence_length
		max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
		max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)

		# x: [batch_size, time_step, embedding_size], float32
		self.x = tf.nn.embedding_lookup(self.embedding, source) # 训练的输入
		# y: [batch_size, time_step]
		self.y = tgt

		cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num) # rnn神经元
		cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
		if DROPOUT_RATE is not None:
			cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
			cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

		# time_major 可以适应输入维度。
		outputs, bi_state = \
			tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32) # 双向rnn层

		forward_out, backward_out = outputs # 网络输出
		outputs = tf.concat([forward_out, backward_out], axis=2)

		# projection: 带权输入和输出，函数映射
		W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
		b = tf.get_variable("projection_b", [TAGS_NUM])
		x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
		projection = tf.matmul(x_reshape, W) + b

		# -1 to time step 网络输出
		self.outputs = tf.reshape(projection, [self.batch_size, -1, TAGS_NUM])

		self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch], dtype=tf.int32)
		self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood( # CRF似然函数
			self.outputs, self.y, self.seq_length)

		# Add a training op to tune the parameters.
		self.loss = tf.reduce_mean(-self.log_likelihood) # 训练的损失函数，即CRF似然函数的相反数
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss) # 训练的op，优化损失函数，梯度下降


def train(net, iterator, sess): # 网络的训练
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(model_path) # 获取已经训练的模型，在此基础之上继续训练
	if ckpt is not None:
		path = ckpt.model_checkpoint_path
		print 'loading pre-trained model from %s.....' % path
		saver.restore(sess, path)

	current_epoch = sess.run(net.global_step) # 当前训练的周期
	while True:
		if current_epoch > EPOCH: break
		try:
			tf_unary_scores, tf_transition_params, _, losses = sess.run(
				[net.outputs, net.transition_params, net.train_op, net.loss])

			if current_epoch % 100 == 0: # 每100个周期输出一次
				print '*' * 100
				print current_epoch, 'loss', losses
				print '*' * 100

			# 每隔10%的进度则save一次。
			if current_epoch % (EPOCH / 10) == 0 and current_epoch != 0:
				sess.run(tf.assign(net.global_step, current_epoch))
				saver.save(sess, model_path + 'points', global_step=current_epoch)

			current_epoch += 1

		except tf.errors.OutOfRangeError:
			sess.run(iterator.initializer)
		except tf.errors.InvalidArgumentError:
			print current_epoch, ' iterator.next() cannot get enough data to a batch, initialize it.'
			sess.run(iterator.initializer)
	print 'training finished!'


def predict(net, tag_table, sess): # 网络的测试
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(model_path) # 读取保存的网络模型
	if ckpt is not None:
		path = ckpt.model_checkpoint_path
		print 'loading pre-trained model from %s.....' % path
		saver.restore(sess, path)
	else:
		print 'Model not found, please train your model first'
		return

	# 获取原文本的iterator
	file_iter = file_content_iterator(pred_file)

	while True:
		# batch等于1的时候本来就没有padding，如果批量预测的话，记得这里需要做长度的截取。
		try:
			tf_unary_scores, tf_transition_params = sess.run(
				[net.outputs, net.transition_params])
		except tf.errors.OutOfRangeError:
			print 'Prediction finished!'
			break

		# 把batch那个维度去掉
		tf_unary_scores = np.squeeze(tf_unary_scores)

		viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
			tf_unary_scores, tf_transition_params)
		tags = [] # 存放预测的实体类别
		for id in viterbi_sequence:
			tags.append(sess.run(tag_table.lookup(tf.constant(id, dtype=tf.int64))))
		write_result_to_file(file_iter, tags) #预测结果写入文件
	# 预测准确率计算
	all = 0
	hit = 0
	all1 = 0
	hit1 = 0
	entity_pre = []
	tag_pre = []
	entity_out = []
	tag_out = []
	with codecs.open('resource/predict.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			entity = []
			for i in range(len(line)):
				entity.append(line[i])
			entity_pre.append(entity)
	with codecs.open('resource/pre_target.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			tag = []
			for i in range(len(line)):
				tag.append(line[i])
			tag_pre.append(tag)
	with codecs.open('resource/output_word.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			entity = []
			for i in range(len(line)):
				entity.append(line[i])
			entity_out.append(entity)
	with codecs.open('resource/output_tag.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			tag = []
			for i in range(len(line)):
				tag.append(line[i])
			tag_out.append(tag)
	if len(entity_pre) == len(entity_out):
		for i in range(len(entity_pre)):
			ite = len(entity_out[i]) if len(entity_out[i]) < len(entity_pre[i]) else len(entity_pre[i])
			for j in range(ite):
				if tag_pre[i][j] == 'O' and tag_out[i][j] == 'O':
					all += 1
					hit += 1
					continue
				else:
					all += 1
					all1 += 1
					if tag_out[i][j] == tag_pre[i][j]:
						hit += 1
						hit1 += 1
	hit = float(hit)
	all = float(all)
	hit1 = float(hit1)
	all1 = float(all1)
	presion = hit/all
	presion1 = hit1/all1
	print '全部预测准确率：' + str(presion)
	print '有效预测准确率：' + str(presion1)



if __name__ == '__main__':

	action = config.FLAGS.action
	# 获取词的总数。
	vocab_size = get_src_vocab_size()
	src_unknown_id = tgt_unknown_id = vocab_size
	src_padding = vocab_size + 1

	src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id,
	                                                        tgt_unknown_id)
	embedding = load_word2vec_embedding(vocab_size)

	if action == 'train':
		iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
	elif action == 'predict':
		BATCH_SIZE = 1
		DROPOUT_RATE = 1.0
		iterator = get_predict_iterator(src_vocab_table, vocab_size, BATCH_SIZE)
	else:
		print 'Only support train and predict actions.'
		exit(0)

	tag_table = tag_to_id_table()
	net = NER_net("ner", iterator, embedding, BATCH_SIZE)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(iterator.initializer)
		tf.tables_initializer().run()

		if action == 'train':
			train(net, iterator, sess)
		elif action == 'predict':

			predict(net, tag_table, sess)
