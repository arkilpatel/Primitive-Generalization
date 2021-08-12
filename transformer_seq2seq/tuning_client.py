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
	config['blank_op'] = temp['blank_op']
	config['dev_only'] = temp['dev_only']
	config['tag_emb'] = temp['tag_emb']
	config['tag_hid'] = temp['tag_hid']
	config['other_inp_name'] = temp['other_inp_name']
	config['epochs'] = temp['epochs']
	config['save_model'] = temp['save_model']
	config['show_train_acc'] = temp['show_train_acc']
	config['embedding'] = temp['embedding']
	config['emb_name'] = temp['emb_name']
	config['freeze_emb'] = temp['freeze_emb']
	config['d_model'] = temp['d_model']
	config['d_model1'] = temp['d_model1']
	config['d_model2'] = temp['d_model2']
	config['d_ff'] = temp['d_ff']
	config['decoder_layers'] = temp['decoder_layers']
	config['encoder_layers'] = temp['encoder_layers']
	config['heads'] = temp['heads']
	config['batch_size'] = temp['batch_size']
	config['lr'] = temp['lr']
	config['emb_lr'] = temp['emb_lr']
	config['dropout'] = temp['dropout']

	cmd = 'python {}'.format(config['src_file'])
	del config['src_file']

	run_name = 'RUN'
	for key, value in config.items():
		if key not in ['mode', 'dev_set', 'test_set', 'show_train_acc', 'emb_name', 'dev_always', 'test_always', 'gen_always', 'dev_only', 'eval_last_n', 'other_inp_name', 'freeze_emb']:
			run_name = run_name + '-{}{}'.format(key.replace('_',''), str(value).replace('.',''))
		if key in ['freeze_emb', 'results', 'debug', 'save_model', 'show_train_acc', 'dev_set', 'test_set', 'dev_always', 'test_always', 'gen_always', 'blank_op',	
					'dev_only', 'tag_emb', 'tag_hid']:
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
			if key == 'blank_op':
				if value == True:
					cmd += ' -blank_op'
				else:
					cmd += ' -no-blank_op'
			if key == 'tag_emb':
				if value == True:
					cmd += ' -tag_emb'
				else:
					cmd += ' -no-tag_emb'
			if key == 'tag_hid':
				if value == True:
					cmd += ' -tag_hid'
				else:
					cmd += ' -no-tag_hid'
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