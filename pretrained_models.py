import os
import csv
import pickle
import sys
import datetime
import tensorflow as tf
#tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
import numpy as np
import scipy.spatial.distance as ds
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from random import randrange
import json
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from nltk.corpus import stopwords
import time

punctuations = '''!()[]{};:'"\,<>./?@#$%^&*_~**'''
letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
intent =[]
snippet =[]
stop_words = set(stopwords.words('english'))
LIM = 100

BASE_PATH = "."
CBOW_PATH = "model_200_W10_CBOW_NEG5.bin"
ELMO_PATH = os.path.join('model')
CBOW_COMPRESSED_PATH = "corpus_book.bin"

def inc(sent):
	if sent=="":
		sent = "None"
	sent = sent.split()
	if len(sent)>LIM:
		sent = sent[:LIM]
	return sent

def preProcess1(inword):
    wrds = inword.split()
    w_final =[]
    for w in wrds:
        w_camel1= re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])', w) 
        w_camel2 = [w.lower() for w in w_camel1]
        w_camel = [w for w in w_camel2 if not w in stop_words]
        for w3 in w_camel:
            w_final.append(w3)
    return w_final    

def preProcess2(final):
    w_final=[]
    for i in range(0, len(final)):
        if final[i] in letters:
            if final[i-1] not in letters:
                w_final.append('variable')
        if final[i] not in letters:
            w_final.append(final[i])
    return w_final

def getActual(text):
	wrd_list_1 = preProcess1(text)
	wrd_list_2 = preProcess2(wrd_list_1)
	return ' '.join(wrd_list_2)

class EpochSaver(CallbackAny2Vec):
	'''Callback to save model after each epoch.'''
	def __init__(self):
		epoch = 0
		cur_time = datetime.datetime.now()

	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(epoch),flush=True)
		cur_time = datetime.datetime.now()

	def on_epoch_end(self, model):
		print("Epoch #{} end".format(epoch),flush=True)
		delta = datetime.datetime.now()-cur_time
		print("Time taken : ",delta,flush=True)
		epoch += 1

def cosine_similarity(data1, data2):
    val = 1 - ds.cosine(data1, data2)
    assert val <= 1
    return val

def getClosestK(points,cur_point):
    all_order = []
    for x in points:
        all_order.append((cosine_similarity(cur_point,x),x))
    allorder = sorted(all_order,key=lambda x: x[0])
    allorder.reverse()
    return allorder

def concat_vector1(list1,list2):
	return np.concatenate([list1,list2],axis=1)

def concat_vector2(list1,list2):
	return np.maximum(list1,list2)	

def normalize_l2(data_part):
	return normalize(data_part, norm="l2")

def avg_vec(vec_list):
	avg = []
	n = len(vec_list)
	for i in range(len(vec_list[0])):
		avg.append(0)
		for j in range(n):
			avg[i] += vec_list[j][i]
		avg[i] /= n

	return normalize_l2([avg])[0]



class embeddingModel:

	def load_CBOW_model(self):
		'''
		Function to load CBOW model
		'''
		model_name = os.path.join(BASE_PATH,CBOW_PATH)
		self.CBOWmodel = Word2Vec.load(model_name)

	def load_CBOW_compressed_model(self):
		'''
		Function to load the compressed CBOW model
		'''
		model_name = os.path.join(BASE_PATH,CBOW_COMPRESSED_PATH)
		self.CBOWmodel = Word2Vec.load(model_name)


	def get_embed_CBOW_word(self,wrd):
		'''
		Function that takes a word as input, and returns the vector corresponding to the word.
		If word is not found, returns a vector with all values equal to 1e-6
		'''
		vec = [1e-6 for i in range(200)]
		if wrd in self.CBOWmodel.wv.vocab:
			vec = self.CBOWmodel[wrd]
		return vec

	def get_embed_CBOW_sent(self,sent):
		'''
		Function that takes a sentence(single string) as input, and returns vectors for each word in the sentence
		'''
		word_list = inc(sent)
		return [self.get_embed_CBOW_word(word) for word in word_list]

	def get_embed_CBOW_sent_avg(self,sent):
		'''
		Function that takes a sentence(single string) as input, and returns a single vector which is the average of vectors of all words in the sentence
		'''
		tokenized_context = sent.split()
		data_part = [1e-9 for i in range(200)]
		cnt = 0
		for wrd in tokenized_context:
			vec_r = self.get_embed_CBOW_word(wrd)
			cnt+=1
			for i in range(200):
				data_part[i]+=vec_r[i]
		if cnt>0:
			for i in range(200):
				data_part[i]/=cnt
		return data_part

	def get_embed_CBOW_wordwise(self,sent):
		'''
		Function that takes a sentence as input, and returns a dictionary with words as the keys, and their vectors as the values
		'''
		tokenized_context = sent.split()
		word_vec_map = dict()
		for wrd in tokenized_context:
			vec_r = self.get_embed_CBOW_word(wrd)
			if wrd not in word_vec_map:
				word_vec_map[wrd] = vec_r				
		return word_vec_map

	def load_ELMO_model(self):
		# Location of pretrained LM.  Here we use the test fixtures.
		datadir = os.path.join(BASE_PATH,ELMO_PATH)
		vocab_file = os.path.join(datadir, 'vocab.txt')
		options_file = os.path.join(datadir, 'options.json')
		weight_file = os.path.join(datadir, 'weights.hdf5')
		# Create a Batcher to map text to character ids.
		self.batcher = Batcher(vocab_file, 50)
		# Input placeholders to the biLM.
		self.context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
		# Build the biLM graph.
		bilm = BidirectionalLanguageModel(options_file, weight_file)
		# Get ops to compute the LM embeddings.

		context_embeddings_op = bilm(self.context_character_ids)
		# Get an op to compute ELMo (weighted average of the internal biLM layers)
		self.elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
		# TF session
		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())
    
	def get_embed_ELMO_sent(self,sent):
		'''
		Function that takes a sentence (string) as input, and returns vectors for all words in the sentence
		'''
		tokenized_context = [inc(sent)]
		# Create batches of data.
		context_ids = self.batcher.batch_sentences(tokenized_context)
		# Compute ELMo representations (here for the input only, for simplicity).
		elmo_context_input_ = self.sess.run(
			self.elmo_context_input['weighted_op'],
			feed_dict={self.context_character_ids: context_ids}
		)
		return elmo_context_input_[0]

	
	def get_embed_ELMO_sent_avg(self,sent):
		'''
		Function that takes a sentence (string) as input, and returns vector for the sentence as average of vectors of all words in the sentence
		'''
		tokenized_context = [inc(sent)]
		# Create batches of data.
		context_ids = self.batcher.batch_sentences(tokenized_context)
		# Compute ELMo representations (here for the input only, for simplicity).
		elmo_context_input_ = self.sess.run(
			self.elmo_context_input['weighted_op'],
			feed_dict={self.context_character_ids: context_ids}
		)

		data_part = [1e-9 for i in range(elmo_context_input_.shape[2])]
		for x in range(elmo_context_input_.shape[1]):
			for y in range(elmo_context_input_.shape[2]):
				data_part[y] += elmo_context_input_[0][x][y]
		if elmo_context_input_.shape[1]>0:
			for y in range(elmo_context_input_.shape[2]):
				data_part[y]/=elmo_context_input_.shape[1]

		return data_part

	def get_embed_ELMO_sent_wordwise(self,sent):
		'''
		Function that takes a sentence (string) as input, and returns mapping from word to their vector
		'''
		tokenized_context = [inc(sent)]
		all_word_vectors = self.get_embed_ELMO_sent(sent)
		word_vec_map = {}
		for i in range(len(tokenized_context[0])):
			wrd = tokenized_context[0][i]
			vec = all_word_vectors[i]
			if wrd not in word_vec_map:
				word_vec_map[wrd] = [vec]
			else:
				word_vec_map[wrd].append(vec)

		# averaging all words

		for wrd in word_vec_map.keys():
			word_vec_map[wrd] = avg_vec(word_vec_map[wrd])

		return word_vec_map

			
	def get_embed_ELMO_CBOW_concat(self,sent):
		elmo_embed = normalize_l2(self.get_embed_ELMO_sent(sent))
		cbow_embed = normalize_l2(self.get_embed_CBOW_sent(sent))
		return concat_vector1(elmo_embed,cbow_embed)

	def get_embed_ELMO_CBOW_concat_avg(self,sent):
		elmo_embed = normalize_l2([self.get_embed_ELMO_sent_avg(sent)])
		cbow_embed = normalize_l2([self.get_embed_CBOW_sent_avg(sent)])
		return concat_vector1(elmo_embed,cbow_embed)[0]

	def get_embed_ELMO_CBOW_concat_wordwise(self,sent):
		'''
		Function that takes a sentence (string) as input, and returns mapping from word to their vector
		'''
		tokenized_context = [inc(sent)]
		all_word_vectors = self.get_embed_ELMO_CBOW_concat(sent)
		word_vec_map = {}
		for i in range(len(tokenized_context[0])):
			wrd = tokenized_context[0][i]
			vec = all_word_vectors[i]
			if wrd not in word_vec_map:
				word_vec_map[wrd] = [vec]
			else:
				word_vec_map[wrd].append(vec)

		# averaging all words

		for wrd in word_vec_map.keys():
			word_vec_map[wrd] = avg_vec(word_vec_map[wrd])

		return word_vec_map



	def get_embed_ELMO_CBOW_max(self,sent):
		elmo_embed = normalize_l2(self.get_embed_ELMO_sent(sent))
		cbow_embed = normalize_l2(self.get_embed_CBOW_sent(sent))
		return concat_vector2(elmo_embed,cbow_embed)

	def get_embed_ELMO_CBOW_max_avg(self,sent):
		elmo_embed = normalize_l2([self.get_embed_ELMO_sent_avg(sent)])
		cbow_embed = normalize_l2([self.get_embed_CBOW_sent_avg(sent)])
		return concat_vector2(elmo_embed,cbow_embed)[0]

	def get_embed_ELMO_CBOW_max_wordwise(self,sent):
		'''
		Function that takes a sentence (string) as input, and returns mapping from word to their vector
		'''
		tokenized_context = [inc(sent)]
		all_word_vectors = self.get_embed_ELMO_CBOW_max(sent)
		word_vec_map = {}
		for i in range(len(tokenized_context[0])):
			wrd = tokenized_context[0][i]
			vec = all_word_vectors[i]
			if wrd not in word_vec_map:
				word_vec_map[wrd] = [vec]
			else:
				word_vec_map[wrd].append(vec)

		# averaging all words

		for wrd in word_vec_map.keys():
			word_vec_map[wrd] = avg_vec(word_vec_map[wrd])

		return word_vec_map	
