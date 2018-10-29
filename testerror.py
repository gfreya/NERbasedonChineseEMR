#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import codecs
import os
import six

class Config:
	data_path_train = '/home/neufxt'
	embedded_vector_final = 'zhvec'
	vec_error = 'zhvecerror'

	data_path = '/home/neufxt/zhwiki'
	zhwiki_bz2 = 'zhwiki-latest-pages-articles.xml.bz2'
	zhwiki_raw = 'zhwiki_raw.txt'
	zhwiki_raw_t2s = 'zhwiki_raw_t2s.txt'
	zhwiki_seg_t2s = 'zhwiki_seg.txt'
	embedded_model_t2s = 'embedding_model_t2s/zhwiki_embedding_t2s.model'
	embedded_vector_t2s = 'embedding_model_t2s/vector_t2s'

#config = Config()

def dataformat(_config, vec_size):
	if six.PY3:
		output = open(os.path.join(_config.data_path_train, _config.vec_error), 'w', encoding='utf-8')
	output = codecs.open(os.path.join(_config.data_path_train, _config.vec_error), 'w', encoding='utf-8')
	print('Start...')
	array = []
	with codecs.open(os.path.join(_config.data_path_train, _config.embedded_vector_final), 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			array = array + line.split()
			if len(array) == vec_size + 1:
				output.write(array[0] + ':')
				for i in range(1, vec_size):
					output.write(array[i] + ',')
				output.write(array[vec_size] + '\n')
				array = []
	output.close()
	print("Finished!")

#dataformat(config, 300)

def test_predict():
	entity = []
	tag = []
	string = ''
	with codecs.open('resource/predict.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			entity.append(len(line))
	with codecs.open('resource/pre_target.txt', 'r', encoding='utf-8') as raw_input:
		for line in raw_input.readlines():
			line = line.strip()
			line = line.split()
			tag.append(len(line))
	for i in range(len(entity)):
		string += str(entity[i]) + ' '
	print(string)
	string = ''
	for i in range(len(tag)):
		string += str(tag[i]) + ' '
	print(string)

test_predict()
