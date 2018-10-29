#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from gensim.corpora import WikiCorpus
import jieba
import codecs
import os
import six
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

import sys
reload(sys)
sys.setdefaultencoding('utf8')

jieba.load_userdict("resource/newdict.txt")

#wiki数据初步处理
class Config:
	data_path_train = '/home/leo/PycharmProject/ChNer'#path
	embedded_vector_final = 'demo/resource/wiki.zh.vec'#output word2vec
	# embedded_vector_final1 = 'demo/resource/wiki.zh0.vec'
	data_source = '/home/leo/train_data600' + '/train_data600'#raw-input-train
	data_predict = '/home/leo/train_data600' + '/test_data600'#raw-input-test
	source = 'demo/resource/source.txt'#output-train
	target = 'demo/resource/target.txt'#output-train-label
	predict = 'demo/resource/predict.txt'#output-test
	pre_target = 'demo/resource/pre_target.txt'#output-test-label
	data_newdict = 'demo/resource/newdict.txt' #output-dict

	data_path = '/home/neufxt/zhwiki'#wiki-path
	zhwiki_bz2 = 'zhwiki-latest-pages-articles.xml.bz2'#wiki-file
	zhwiki_raw = 'zhwiki_raw.txt'#temp file
	zhwiki_raw_t2s = 'zhwiki_raw_t2s.txt'#temp file2
	zhwiki_seg_t2s = 'zhwiki_seg.txt'#wiki-seg-file

	embedded_model_t2s = 'embedding_model_t2s/zhwiki_embedding_t2s.model'#词向量模型
	embedded_vector_t2s = 'embedding_model_t2s/vector_t2s'#词向量文件



#处理维基百科文章，得到结果文件zhwiki_raw.txt
#结果文件里包含繁体文字，（在控制台通过OpenCC），把文本内的繁体文字转化成简体文字，得到文件zhwiki_raw_t2s
def dataprocess(_config):
	i = 0
	if six.PY3:
		output = open(os.path.join(_config.data_path, _config.zhwiki_raw), 'w')
	output = codecs.open(os.path.join(_config.data_path, _config.zhwiki_raw), 'w')
	wiki = WikiCorpus(os.path.join(_config.data_path, _config.zhwiki_bz2), lemmatize=False, dictionary={})
	for text in wiki.get_texts():
		if six.PY3:
			output.write(b' '.join(text).decode('utf-8', 'ignore') + '\n')
		else:
			output.write(' '.join(text) + '\n')
		i += 1
		if i % 10000 == 0:
			print('Saved ' + str(i) + ' articles')
	output.close()
	print('Finished Saved ' + str(i) + ' articles')


config = Config()
#dataprocess(config)


def is_alpha(tok):
	try:
		return tok.encode('ascii').isalpha()
	except UnicodeEncodeError:
		return False

#文本分词，结果文件zhwiki_seg.txt
def zhwiki_segment(_config, remove_alpha=True):
	i = 0
	if six.PY3:
		output = open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path, _config.zhwiki_seg_t2s), 'w', encoding='utf-8')
	print('Start...')
	with codecs.open(os.path.join(_config.data_path, _config.zhwiki_raw_t2s), 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			i += 1
			print('line ' + str(i))
			text = line.split()
			if True:
				text = [w for w in text if not is_alpha(w)]
			word_cut_seed = [jieba.cut(t) for t in text]
			tmp = ''
			for sent in word_cut_seed:
				for tok in sent:
					tmp += tok + ' '
			tmp = tmp.strip()
			if tmp:
				output.write(tmp + '\n')
		output.close()

#zhwiki_segment(config)

#训练word2Vec词向量
def word2vec(_config, saved=False):
	print('Start...')
	model = Word2Vec(LineSentence(os.path.join(_config.data_path, _config.zhwiki_seg_t2s)),
					 size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
	if saved:
		model.save(os.path.join(_config.data_path, _config.embedded_model_t2s))
		#model.save_word2vec_format(os.path.join(_config.data_path, _config.embedded_vector_t2s), binary=False)
		model.wv.save_word2vec_format(os.path.join(_config.data_path, _config.embedded_vector_t2s), binary=False)
	print("Finished!")
	return model

#利用词向量计算词相似度
def wordsimilarity(word, model):
	semi = ''
	try:
		semi = model.most_similar(word, topn=10)
	except KeyError:
		print('The word not in vocabulary!')
	for term in semi:
		print('%s,%s' % (term[0], term[1]))


#model = word2vec(config, saved=True)

#将已经训练好的词向量数据调整成指定格式
def dataformat(_config, vec_size):
	if six.PY3:
		output = open(os.path.join(_config.data_path_train, _config.embedded_vector_final), 'w', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path_train, _config.embedded_vector_final), 'w', encoding='utf-8')
	print('Start...')
	array = []
	with codecs.open(os.path.join(_config.data_path, _config.embedded_vector_t2s), 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			array = array + line.split()
			if len(array) == vec_size + 1:
				#output.write(array[0] + ':')
				for i in range(0, vec_size):
					output.write(array[i] + ' ')
				output.write(array[vec_size] + '\n')
				array = []
	output.close()
	print("Finished!")

#dataformat(config, 300)

#no-use
# def replace(_config):
# 	if six.PY3:
# 		output = open(os.path.join(_config.data_path_train, _config.embedded_vector_final1), 'w', encoding='utf-8')
# 	output = codecs.open(os.path.join(_config.data_path_train, _config.embedded_vector_final1), 'w', encoding='utf-8')
# 	print('Start...')
# 	with codecs.open(os.path.join(_config.data_path_train, _config.embedded_vector_final), 'r', encoding='utf-8') as raw_input:
# 		for line in raw_input.readlines():
# 			line = str(line).replace(':',' ')
# 			line.replace(',',' ')
# 			output.write(line)
# 	output.close()
# 	print("Finished!")

#replace(config)

#无用
# def fileformat(_config):
# 	if six.PY3:
# 		#output_source = open(os.path.join(_config.data_path_train, _config.source), 'a', encoding='utf-8')
# 		#output_target = open(os.path.join(_config.data_path_train, _config.target), 'a', encoding='utf-8')
# 		output_predict = open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
# 	#output_source = codecs.open(os.path.join(_config.data_path_train, _config.source), 'a', encoding='utf-8')
# 	#output_target = codecs.open(os.path.join(_config.data_path_train, _config.target), 'a', encoding='utf-8')
# 	output_predict = codecs.open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
# 	print('Start...')
# 	#files = os.listdir(_config.data_source)
# 	files = os.listdir(_config.data_predict)
# 	for file in files:
# 		if not('txtoriginal' in file):
# 			#with codecs.open(os.path.join(_config.data_source, file), 'r', encoding='utf-8') as raw_input:
# 			with codecs.open(os.path.join(_config.data_predict, file), 'r', encoding='utf-8') as raw_input:
# 				for line in raw_input.readlines():
# 					line = line.strip()
# 					line = line.split()
# 					#output_source.write(line[0] + ' ')
# 					output_predict.write(line[0] + ' ')
# 					#症状和体征(symptom_sign),检查和检验(check_test),疾病和诊断(disease_diagnose),治疗(cure),身体部位(bodypart)
# 					# if line[3] == u'症状和体征':
# 					# 	output_target.write('symptom_sign ')
# 					# elif line[3] == u'检查和检验':
# 					# 	output_target.write('check_test ')
# 					# elif line[3] == u'疾病和诊断':
# 					# 	output_target.write('disease_diagnose ')
# 					# elif line[3] == u'治疗':
# 					# 	output_target.write('cure ')
# 					# elif line[3] == u'身体部位':
# 					# 	output_target.write('bodypart ')
# 	#output_source.close()
# 	#output_target.close()
# 	output_predict.close()
# 	print("Finished!")

#fileformat(config)

#分词用的自定义词典，来自training dataset 1-200文件夹里的识别完实体的结果文件
def train_newdict(_config):
	if six.PY3:
		output_newdict = open(os.path.join(_config.data_path_train, _config.data_newdict), 'a', encoding='utf-8')
	output_newdict = codecs.open(os.path.join(_config.data_path_train, _config.data_newdict), 'a', encoding='utf-8')
	print('Start...')
	files = os.listdir(_config.data_source)
	for file in files:
		if not('txtoriginal' in file):
			with codecs.open(os.path.join(_config.data_source, file), 'r', encoding='utf-8') as raw_input:
				for line in raw_input.readlines():
					line = line.strip()
					line = line.split()
					flag = True
					with codecs.open(os.path.join(_config.data_path_train, _config.data_newdict), 'r', encoding='utf-8') as raw_input1:
						for line1 in raw_input1.readlines():
							if line[0] in line1:
								flag = False
								break
					if flag:
						output_newdict.write(line[0] + '\n')
						#jieba.suggest_freq(line[0], True)
	output_newdict.close()
	print("Finished!")

#train_newdict(config)

def convert(str):
	#症状和体征(symptom_sign),检查和检验(check_test),疾病和诊断(disease_diagnose),治疗(cure),身体部位(bodypart)
	# if str == u'症状和体征':
	# 	return 'symptom_sign'
	# elif str == u'检查和检验':
	# 	return 'check_test'
	# elif str == u'疾病和诊断':
	# 	return 'disease_diagnose'
	# elif str == u'治疗':
	# 	return 'cure '
	# elif str == u'身体部位':
	# 	return 'bodypart'


	if str == u'解剖部位':
		return 'part'
	elif str == u'症状描述':
		return 'symptom'
	elif str == u'独立症状':
		return 'ind_symp'
	elif str == u'药物':
		return 'medication '
	elif str == u'手术':
		return 'surgery'


# 1. 解剖部位：由多种组织构成能行使功能的结构单位，例如“腹部”。
# 2. 症状描述：指患者患病后对机体生理功能异常的自身体验和感觉，同时需与解剖部位分开输出，例如“腹部不适”，需分别输出“腹部”、 “不适”。
# 3. 独立症状：指患者患病后对机体生理功能异的自身体验和感觉，可作为独立症状词输出，例如“眩晕”。
# 4. 药物：用来治疗、预防或促进健康的一种化学物质。
# 5. 手术：指医生用医疗器械对病人身体进行的切除、缝合等治疗。

#处理程序的输入文件格式(处理training dataset 1-200文件夹里的数据，分词、过滤、给实体标记类别)，生成source.txt和target.txt
#同时此函数也可用于处理测试文件
def train_segment(_config, remove_alpha=True):
	#i = 0
	if six.PY3:
		output = open(os.path.join(_config.data_path_train, _config.source), 'a', encoding='utf-8')
		#output = open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
		output_target = open(os.path.join(_config.data_path_train, _config.target), 'a', encoding='utf-8')
		#output_target = open(os.path.join(_config.data_path_train, _config.pre_target), 'a', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path_train, _config.source), 'a', encoding='utf-8')
	#output = codecs.open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
	output_target = codecs.open(os.path.join(_config.data_path_train, _config.target), 'a', encoding='utf-8')
	#output_target = codecs.open(os.path.join(_config.data_path_train, _config.pre_target), 'a', encoding='utf-8')
	print('Start_train_segment...')
	files = os.listdir(_config.data_source)
	#files = os.listdir(_config.data_predict)
	dict = {}
	for file in files:
		if 'txtoriginal' in file:
			with codecs.open(os.path.join(_config.data_source, file), 'r', encoding='utf-8') as raw_input:
			#with codecs.open(os.path.join(_config.data_predict, file), 'r', encoding='utf-8') as raw_input:
				file_target = file.replace('t', '')
				file_target = file_target.replace('x','')
				file_target = file_target.replace( 'o', '')
				file_target = file_target.replace( 'r', '')
				file_target = file_target.replace( 'i', '')
				file_target = file_target.replace( 'g', '')
				file_target = file_target.replace( 'n', '')
				file_target = file_target.replace( 'a', '')
				file_target = file_target.replace( 'l', '')
				file_target = file_target.replace( '.', '')
				file_target = file_target + '.txt'
				with codecs.open(os.path.join(_config.data_source, file_target), 'r', encoding='utf-8') as raw_input1:
				#with codecs.open(os.path.join(_config.data_predict, file_target), 'r', encoding='utf-8') as raw_input1:
					for line in raw_input1.readlines():
						line = line.strip()
						line = line.split()
						dict[line[0]] = line[3]
						#print(file_target+line[0]+line[3]+dict[line[0]])
				for line1 in raw_input.readlines():
					line1 = line1.strip()
					#i += 1
					#print('line ' + str(i))
					text = line1.split()
					if True:
						text = [w for w in text if not is_alpha(w)]
					word_cut_seed = [jieba.cut(t) for t in text]
					tmp = ''
					tmp1 = ''
					for sent in word_cut_seed:
						for tok in sent:
							if dict.has_key(tok):
								#tmp1 += convert(dict[tok]) + ' '
								#output_target.write(dict[tok] + ' ')
								if dict[tok] == u'解剖部位':
									tmp1 += 'part '
								elif dict[tok] == u'症状描述':
									tmp1 += 'symptom '
								elif dict[tok] == u'独立症状':
									tmp1 += 'ind_symp '
								elif dict[tok] == u'药物':
									tmp1 += 'medication '
								elif dict[tok] == u'手术':
									tmp1 += 'surgery '
							else:
								tmp1 += 'O '
								#output_target.write('O ')
							tmp += tok + ' '
					tmp = tmp.strip()
					tmp1 = tmp1.strip()
					if tmp:
						output.write(tmp + '\n')
					if tmp1:
						output_target.write(tmp1 + '\n')
			dict = {}
	output.close()
	output_target.close()

#train_segment(config)

#处理程序的输入文件格式(处理training dataset 1-200文件夹里的数据，分词、过滤、给实体标记类别)，生成source.txt和target.txt
#同时此函数也可用于处理测试文件
def test_segment(_config, remove_alpha=True):
	#i = 0
	if six.PY3:
		output = open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
		output_target = open(os.path.join(_config.data_path_train, _config.pre_target), 'a', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
	output_target = codecs.open(os.path.join(_config.data_path_train, _config.pre_target), 'a', encoding='utf-8')
	print('Start_test_segment...')
	files = os.listdir(_config.data_predict)
	dict = {}
	for file in files:
		if 'txtoriginal' in file:
			with codecs.open(os.path.join(_config.data_predict, file), 'r', encoding='utf-8') as raw_input:
				file_target = file.replace('t', '')
				file_target = file_target.replace('x','')
				file_target = file_target.replace( 'o', '')
				file_target = file_target.replace( 'r', '')
				file_target = file_target.replace( 'i', '')
				file_target = file_target.replace( 'g', '')
				file_target = file_target.replace( 'n', '')
				file_target = file_target.replace( 'a', '')
				file_target = file_target.replace( 'l', '')
				file_target = file_target.replace( '.', '')
				file_target = file_target + '.txt'
				with codecs.open(os.path.join(_config.data_predict, file_target), 'r', encoding='utf-8') as raw_input1:
					for line in raw_input1.readlines():
						line = line.strip()
						line = line.split()
						dict[line[0]] = line[3]
						#print(file_target+line[0]+line[3]+dict[line[0]])
				for line1 in raw_input.readlines():
					line1 = line1.strip()
					#i += 1
					#print('line ' + str(i))
					text = line1.split()
					if True:
						text = [w for w in text if not is_alpha(w)]
					word_cut_seed = [jieba.cut(t) for t in text]
					tmp = ''
					tmp1 = ''
					for sent in word_cut_seed:
						for tok in sent:
							if dict.has_key(tok):
								#tmp1 += convert(dict[tok]) + ' '
								#output_target.write(dict[tok] + ' ')
								if dict[tok] == u'解剖部位':
									tmp1 += 'part '
								elif dict[tok] == u'症状描述':
									tmp1 += 'symptom '
								elif dict[tok] == u'独立症状':
									tmp1 += 'ind_symp '
								elif dict[tok] == u'药物':
									tmp1 += 'medication '
								elif dict[tok] == u'手术':
									tmp1 += 'surgery '
							else:
								tmp1 += 'O '
								#output_target.write('O ')
							tmp += tok + ' '
					tmp = tmp.strip()
					tmp1 = tmp1.strip()
					if tmp:
						output.write(tmp + '\n')
					if tmp1:
						output_target.write(tmp1 + '\n')
			dict = {}
	output.close()
	output_target.close()

test_segment(config)

#预测数据的处理，同训练数据，但是不标注实体类别
def data_predict(_config):
	if six.PY3:
		output = open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path_train, _config.predict), 'a', encoding='utf-8')
	print('Start...')
	# with codecs.open(_config.data_newdict, 'r', encoding='utf-8') as raw_input1:
	# 	for line1 in raw_input1.readlines():
	# 		line1 = line1.strip()
	# 		jieba.suggest_freq(line1, True)

	files = os.listdir(_config.data_predict)
	for file in files:
		if 'txtoriginal' in file:
			with codecs.open(os.path.join(_config.data_predict, file), 'r', encoding='utf-8') as raw_input:
				for line in raw_input.readlines():
					line = line.strip()
					text = line.split()
					if True:
						text = [w for w in text if not is_alpha(w)]
					word_cut_seed = [jieba.cut(t) for t in text]
					tmp = ''
					for sent in word_cut_seed:
						for tok in sent:
							tmp += tok + ' '
					tmp = tmp.strip()
					if tmp:
						output.write(tmp + '\n')
	output.close()
	print("Finished!")

#data_predict(config)
