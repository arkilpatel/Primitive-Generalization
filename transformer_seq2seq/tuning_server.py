import os
import sys
import pdb
import numpy as np
import random
import argparse
import codecs
import pickle
import time
import json
# sys.path.insert(0, '/mnt/kabir/services/queue/')
sys.path.insert(0, '/datadrive/global_files/queue/')

from queue_client import QueueClient
from collections import defaultdict as ddict
from collections import OrderedDict
from pprint import pprint

parser = argparse.ArgumentParser(description='')
parser.add_argument('-port',        required=True)
parser.add_argument('-clear',     	action='store_true')
parser.add_argument('-allclear',    action='store_true')
args = parser.parse_args()

q = QueueClient('http://0.0.0.0:{}/'.format(args.port))
if args.clear:
	q.clear()
if args.allclear:
	q.clear()
	exit(0)

exclude_ids = set([])

src_file = "-m src.main"

_emb_lr = [0.0005] # Same as LR ###############################
_batch_size = [1, 2]
_lr = [0.0008, 0.001]
_d_ff = [16, 32]
_d_model = [16, 32]
_d_model1 = [128] # [128, 256]
_d_model2 = [128] # [128, 256]
_encoder_layers = [2, 3]
_decoder_layers = [2] # Same as Encoder layers
_heads = [4, 8]
_dropout = [0.1]

mode = 'train'
dev_set = False
test_set = False
dev_always = False
test_always = False
gen_always = True
dev_only = False
tag_emb = False
tag_hid = False
other_inp_name = "Original\ Input"
eval_last_n = 1
blank_op = False
dataset = 'colors'
epochs = 100

embedding = 'random'
emb_name = 'roberta-base'
freeze_emb = False

show_train_acc = False
save_model = False

# opt = 'adam'
i, count = 0, 0

for dropout in _dropout:
	for d_model in _d_model:
		for d_model1 in _d_model1:
			for d_model2 in _d_model2:
				for d_ff in _d_ff:
					for decoder_layers in _decoder_layers:
						for encoder_layers in _encoder_layers:
							for heads in _heads:
								for batch_size in _batch_size:
									for lr in _lr:
										for emb_lr in _emb_lr:
											config = OrderedDict()
											config['src_file'] = src_file
											config['mode'] = mode
											config['dev_set'] = dev_set
											config['test_set'] = test_set
											config['dev_always'] = dev_always
											config['test_always'] = test_always
											config['gen_always'] = gen_always
											config['eval_last_n'] = eval_last_n
											config['blank_op'] = blank_op
											config['dev_only'] = dev_only
											config['tag_emb'] = tag_emb
											config['tag_hid'] = tag_hid
											config['other_inp_name'] = other_inp_name
											config['dataset'] = dataset
											config['epochs'] = epochs
											config['save_model'] = save_model
											config['show_train_acc'] = show_train_acc
											config['embedding'] = embedding
											config['emb_name'] = emb_name
											config['freeze_emb'] = freeze_emb
											config['d_model'] = d_model
											config['d_model1'] = d_model1
											config['d_model2'] = d_model2
											config['d_ff'] = d_ff
											config['encoder_layers'] = encoder_layers
											config['decoder_layers'] = encoder_layers ###########################
											config['heads'] = heads
											config['batch_size'] = batch_size
											config['lr'] = lr
											config['emb_lr'] = lr
											config['dropout'] = dropout
											if i not in exclude_ids:
												count += 1
												q.enqueue(config)
												print("Inserting {}".format(count), end='\r')

											i += 1

print('\nInserted {}, Total {} in queue. Complete'.format(count, q.getSize()))