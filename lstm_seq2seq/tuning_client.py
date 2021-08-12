# Author : Rishabh Joshi
# Insti  : IISc, Bangalore
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
import subprocess
# sys.path.insert(0, '/mnt/kabir/services/queue/')
sys.path.insert(0, '/datadrive/global_files/queue/')
from queue_client import QueueClient
from collections import defaultdict as ddict
from collections import OrderedDict
from pprint import pprint
# import GPUtil as gputil
thread_num = None

try:
	from subprocess import DEVNULL
except ImportError:
	import os
	DEVNULL = open(os.devnull, 'wb')

def get_cmd(q):
	temp = q.dequeServer()
	if temp == -1:
		print('All Jobs Over!!!!')
		exit(0)
	config = OrderedDict()
	config['src_file'] = temp['src_file']
	config['mode'] = temp['mode']
	config['dev_set'] = temp['dev_set']
	config['test_set'] = temp['test_set']
	config['dataset'] = temp['dataset']
	config['dev_always'] = temp['dev_always']
	config['test_always'] = temp['test_always']
	config['gen_always'] = temp['gen_always']
	config['eval_last_n'] = temp['eval_last_n']
	config['dev_only'] = temp['dev_only']
	config['epochs'] = temp['epochs']
	config['save_model'] = temp['save_model']
	config['show_train_acc'] = temp['show_train_acc']
	config['bidirectional'] = temp['bidirectional']
	config['embedding'] = temp['embedding']
	config['emb_name'] = temp['emb_name']
	config['emb1_size'] = temp['emb1_size']
	config['emb2_size'] = temp['emb2_size']
	config['hidden_size'] = temp['hidden_size']
	config['depth'] = temp['depth']
	config['batch_size'] = temp['batch_size']
	config['lr'] = temp['lr']
	config['emb_lr'] = temp['emb_lr']
	config['dropout'] = temp['dropout']

	cmd = 'python {}'.format(config['src_file'])
	del config['src_file']

	run_name = 'RUN'
	for key, value in config.items():
		if key not in ['mode', 'dev_set', 'test_set', 'save_model', 'show_train_acc', 'emb_name', 'dev_always', 'test_always', 'gen_always', 'dev_only']:
			run_name = run_name + '-{}{}'.format(key.replace('_',''), str(value).replace('.',''))
		if key in ['bidirectional', 'freeze_emb', 'results', 'use_attn', 'debug', 'save_model', 'show_train_acc', 'dev_set', 'test_set', 'dev_always', 'test_always', 'gen_always', 'dev_only']:
			if key == 'bidirectional':
				if value == True:
					cmd += ' -bidirectional'
				else:
					cmd += ' -no-bidirectional'
			if key == 'freeze_emb':
				if value == True:
					cmd += ' -freeze_emb'
				else:
					cmd += ' -no-freeze_emb'
			if key == 'results':
				if value == True:
					cmd += ' -results'
				else:
					cmd += ' -no-results'
			if key == 'use_attn':
				if value == True:
					cmd += ' -use_attn'
				else:
					cmd += ' -no-attn'
			if key == 'debug':
				if value == True:
					cmd += ' -debug'
				else:
					cmd += ' -no-debug'
			if key == 'save_model':
				if value == True:
					cmd += ' -save_model'
				else:
					cmd += ' -no-save_model'
			if key == 'show_train_acc':
				if value == True:
					cmd += ' -show_train_acc'
				else:
					cmd += ' -no-show_train_acc'
			if key == 'dev_set':
				if value == True:
					cmd += ' -dev_set'
				else:
					cmd += ' -no-dev_set'
			if key == 'test_set':
				if value == True:
					cmd += ' -test_set'
				else:
					cmd += ' -no-test_set'
			if key == 'dev_always':
				if value == True:
					cmd += ' -dev_always'
				else:
					cmd += ' -no-dev_always'
			if key == 'test_always':
				if value == True:
					cmd += ' -test_always'
				else:
					cmd += ' -no-test_always'
			if key == 'gen_always':
				if value == True:
					cmd += ' -gen_always'
				else:
					cmd += ' -no-gen_always'
			if key == 'dev_only':
				if value == True:
					cmd += ' -dev_only'
				else:
					cmd += ' -no-dev_only'
		else:
			cmd += ' -{} {}'.format(key, str(value))

	cmd += ' -run_name {}'.format(run_name)
	return cmd

def gpu_run(q, gpu):
	while True:
		cmd = get_cmd(q)
		cmd += ' -gpu ' + str(gpu)
		print('Command: {}'.format(cmd))
		os.system(cmd)

# def cpu_run(q):
# 	while True:
# 		cmd = '. /scratchd/home/ashutosh/environs/synparcpu/bin/activate;' + get_cmd(q)
# 		print('Command: {}'.format(cmd))
# 		os.system(cmd)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Model Tuner')
	parser.add_argument('-gpu',    default='0')
	parser.add_argument('-port',    required=True)
	parser.add_argument('-cpu',    action='store_true')
	parser.add_argument('-batch',  default=None, 	type=int)
	parser.add_argument('-embed',  default=None, 	type=int)
	args = parser.parse_args()
	batch_size = args.batch
	embed = args.embed
	q = QueueClient('http://0.0.0.0:{}/'.format(args.port))

	if args.cpu:
		cpu_run(q)
	else:
		gpu_run(q, args.gpu)