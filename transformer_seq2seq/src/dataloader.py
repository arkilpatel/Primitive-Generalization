import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict

class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['Question', 'Equation', 'Numbers', 'Answer']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder ==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

	def __init__(self, data_path='./data/', dataset='cogs', datatype='train', max_length=60, is_debug=False, to_sort=False, get_tag=False, other_inp_name="Tag", get_blank=False):
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.tsv')
		elif datatype=='dev':
			file_path = os.path.join(data_path, dataset, 'train.tsv')
		elif datatype=='test':
			file_path = os.path.join(data_path, dataset, 'train.tsv')
		else:
			file_path = os.path.join(data_path, dataset, 'gen.tsv')

		self.datatype = datatype
		self.get_tag = get_tag

		file_df= pd.read_csv(file_path, sep='\t')

		self.src= file_df['Input'].values

		if get_blank:
			self.trg= file_df['Blank'].values
		else:
			self.trg= file_df['Output'].values

		if get_tag:
			self.tag= file_df[other_inp_name].values

		if is_debug:
			self.src= self.src[:5000:500]
			self.trg= self.trg[:5000:500]
			if get_tag:
				self.tag= self.tag[:5000:500]

		self.max_length= max_length

		if get_tag:
			all_sents = zip(self.src, self.trg, self.tag)
		else:
			all_sents = zip(self.src, self.trg)

		if to_sort:
			all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))		

		if get_tag:
			self.src, self.trg, self.tag = zip(*all_sents)
		else:
			self.src, self.trg = zip(*all_sents)

	def __len__(self):
		return len(self.src)

	def __getitem__(self, idx):
		src = self.process_string(str(self.src[idx]))
		trg = self.process_string(str(self.trg[idx]))

		if self.get_tag:
			tag = self.process_string(str(self.tag[idx]))
			return {'src': src, 'trg': trg, 'tag': tag}
		return {'src': src, 'trg': trg}

	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		#string = re.sub(r",", " , ", string)
		#string = re.sub(r"!", " ! ", string)
		#string = re.sub(r"\(", " ( ", string)
		#string = re.sub(r"\)", " ) ", string)
		#string = re.sub(r"\?", " ? ", string)
		#string = re.sub(r"\s{2,}", " ", string)
		return string