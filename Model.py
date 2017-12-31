# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:13:03 2017

@author: sabyasachi.ch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score

# Hyper Parameter For Tuning CNN Model and Generating CNN Model
FLAGS = None
MAX_DOCUMENT_LENGTH = 80
EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0
MAX_LABEL = 4
WORDS_FEATURE = 'words'


def cnn_model(features, labels, mode):
    # Layer 2 is basically used to predict words to class.
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  word_vectors = tf.expand_dims(word_vectors, 3)
  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu) #RELU is used as the Activation Function
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID')
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  global n_words
  ###########################################################
  main_data=pandas.read_csv(r"C:\Users\sabyasachi.ch\Desktop\Tensor Text-Tag\CNN.csv",sep=',',encoding="cp1252",error_bad_lines=False,low_memory = False)
  main_data['Text']=main_data['Text'].map(lambda x: x.lstrip('0\thealth\t"').rstrip('"'))
  main_data['Text']=main_data['Text'].map(lambda x: x.lstrip('4\tfinance\t"').rstrip('"'))
  main_data['Text']=main_data['Text'].map(lambda x: x.lstrip('1\tsports\t"').rstrip('"'))
  main_data['Text']=main_data['Text'].map(lambda x: x.lstrip('2\tmovie\t"').rstrip('"'))
  features=main_data['Text']
  labels=main_data['Class']
  train_features , test_features, train_labels, test_labels = train_test_split(features,labels, test_size= 0.2)
  
  x_train = pandas.Series(train_features)
  y_train = pandas.Series(train_labels)
  x_test = pandas.Series(test_features)
  y_test = pandas.Series(test_labels)
  #data['result'] = data['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
  ##########################################################

  # Process vocabulary
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(vocab_processor.fit_transform(x_train)))
  x_test = np.array(list(vocab_processor.transform(x_test)))
  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)


  classifier = tf.estimator.Estimator(model_fn=cnn_model)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=len(x_train),
      num_epochs=None,
      shuffle=False)
  classifier.train(input_fn=train_input_fn, steps=100)

  
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False)

  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
