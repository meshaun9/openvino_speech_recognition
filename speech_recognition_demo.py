from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import cv2
import numpy as np
import mraa
import time

from openvino.inference_engine import IENetwork, IEPlugin

FLAGS = None
UNKNOWN_WORD_LABEL = '_unknown_'
SILENCE_LABEL = '_silence_'
	

def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def create_decoder_graph(): #may need to pass in session 
	"""Creates the input of the CNN model based off of this paper https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
	
	Returns:
	  input node and output node
	"""
	
	
	words_list = prepare_words_list(FLAGS.wanted_words.split(',')) 
	model_settings = prepare_model_settings(
	  len(words_list), FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
	  FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
	runtime_settings = {'clip_stride_ms': FLAGS.clip_stride_ms}


	wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
	decoded_sample_data = contrib_audio.decode_wav(
		wav_data_placeholder,
		desired_channels=1,
		desired_samples=model_settings['desired_samples'],
		name='decoded_sample_data')
	spectrogram = contrib_audio.audio_spectrogram(
		decoded_sample_data.audio,
		window_size=model_settings['window_size_samples'],
		stride=model_settings['window_stride_samples'],
		magnitude_squared=True)
	fingerprint_input = contrib_audio.mfcc(
		spectrogram,
		decoded_sample_data.sample_rate,
		dct_coefficient_count=FLAGS.dct_coefficient_count)
	fingerprint_frequency_size = model_settings['dct_coefficient_count']
	fingerprint_time_size = model_settings['spectrogram_length']
	reshaped_input = tf.reshape(fingerprint_input, [
	  -1, fingerprint_time_size * fingerprint_frequency_size 
		])

	input_frequency_size = model_settings['dct_coefficient_count']
	input_time_size = model_settings['spectrogram_length']
	fingerprint_4d = tf.reshape(reshaped_input,
		[-1, input_time_size, input_frequency_size, 1])
	return wav_data_placeholder,fingerprint_4d

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }

def prepare_inference_engine():
	"""Takes and reads IR(.xml+.bin) from command line,loads device to plugin,
	initializes input and output blobs, and loads the network to the plugin.

	Returns:
	  pointers to the loaded network, input of the network, and output of the network
	"""

	plugin = IEPlugin(device=FLAGS.d, plugin_dirs=FLAGS.plugin_dirs)

	model_xml = FLAGS.m
	model_bin = os.path.splitext(model_xml)[0] + ".bin"

	net = IENetwork.from_ir(model=model_xml, weights=model_bin)

	input_blob = next(iter(net.inputs))# grap inputs shape and dimensions 
	output_blob = next(iter(net.outputs)) # grab output shape and dimension

	plugin = IEPlugin(device=FLAGS.d, plugin_dirs=None)

	net.batch_size = 1 #hardcoding to 1 for now

	exec_net = plugin.load(network=net)

	return exec_net,input_blob,output_blob

def post_processing(results):
	"""Iterates through the output data and displays the results of inference
	
	Arg: results: Resulting ouput data from inference.

	Returns:
	  pointers to the loaded network, input of the network, and output of the network
	"""

	labels =list([line.rstrip() for line in tf.gfile.GFile(FLAGS.labels)])
	for i, probs in enumerate(results):
		top_k = (-results).argsort()
		for node_id in top_k:
			for i in range(len(node_id)):
				human_string = labels[node_id[i]]
				score = probs[node_id[i]]
				print('%s (score = %.5f)' % (human_string, score))
	print("***------------------------------------------------***")

def main(_):
	
	wav_data_placeholder,fingerprint_4d = create_decoder_graph()
	network,input_blob,output_blob = prepare_inference_engine()

	while(True):
		for i in FLAGS.i:

			sess = tf.InteractiveSession()

			with open(i, 'rb') as wav_file: 
				wav_data = wav_file.read()

			input_reshape_data = sess.run(fingerprint_4d, feed_dict= {wav_data_placeholder: wav_data}) 

			outputs = network.infer(inputs={input_blob: input_reshape_data})

			post_processing(outputs[output_blob])

			time.sleep(5)

			sess.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '-sample_rate',
	  type=int,
	  default=16000,
	  help='Expected sample rate of the wavs',)
	parser.add_argument(
	  '-clip_duration_ms',
	  type=int,
	  default=1000,
	  help='Expected duration in milliseconds of the wavs',)
	parser.add_argument(
	  '-clip_stride_ms',
	  type=int,
	  default=30,
	  help='How often to run recognition. Useful for models with cache.',)
	parser.add_argument(
	  '-window_size_ms',
	  type=float,
	  default=30.0,
	  help='How long each spectrogram timeslice is',)
	parser.add_argument(
	  '-window_stride_ms',
	  type=float,
	  default=10.0,
	  help='How long the stride is between spectrogram timeslices',)
	parser.add_argument(
	  '-dct_coefficient_count',
	  type=int,
	  default=40, 
	  help='How many bins to use for the MFCC fingerprint',)
	parser.add_argument(
	  '-i',
	  nargs= '+',
	  default=[], 
	  required = True, 
	  help='What input audio file to use')
	parser.add_argument(
	  '-plugin_dirs',
	   type=str,
	   default = '/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64',
	   help ='Path to directory where plugin library files reside')
	parser.add_argument(
	  '-labels',
	  type=str,
	  default='/home/moniques-robot/Downloads/conv_labels.txt', 
	  help='What input audio file to use')
	parser.add_argument(
	  '-m',
	  type=str,
	  default='',
	  required = True,
	  help='What model architecture to use')
	parser.add_argument(
	  '-wanted_words',
	  type=str,
	  default='yes,no,up,down,left,right,on,off,stop,go',
	  help='Words to use (others will be added to an unknown label)',)
	parser.add_argument(
	  '-d', type=str, default="CPU", help='Device to deploy application on.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]]+ unparsed)

	
