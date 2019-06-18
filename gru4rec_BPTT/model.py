# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq
import numpy as np

class GRU4Rec:
    def __init__(self, args):
        self.args = args
        if not args.is_training:
            self.args.batch_size = 1
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    def build_model(self):
        # input X and target Y, last state of last batch, lengths of sessions in current batch.
        self.X = tf.placeholder(tf.int32, [None, None], name='input')
        self.Y = tf.placeholder(tf.int32, [None, None], name='output')
        self.sess_len = tf.count_nonzero(self.X, 1)
        self.mask = tf.reshape(tf.to_float(tf.not_equal(self.X, 0)), (-1,))
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('LSTM_layer'):
            sigma = self.args.sigma if self.args.sigma != 0 else np.sqrt(6.0 / (self.args.n_items + self.args.rnn_size))
            if self.args.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.args.n_items + 1, self.args.rnn_size])
            softmax_W = tf.get_variable('softmax_w', [self.args.rnn_size, 1 + self.args.n_items])
            softmax_b = tf.get_variable('softmax_b', [self.args.n_items + 1])

            cells = []
            for _ in range(self.args.layers):
                cell = rnn.BasicLSTMCell(self.args.rnn_size, activation=self.hidden_act)
                if self.args.is_training and (self.args.keep_prob < 1.0):
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=self.args.keep_prob)
                cells.append(cell)
            self.cell = cell = rnn.MultiRNNCell(cells)


            zero_state = cell.zero_state(self.args.batch_size, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sess_len, initial_state=zero_state)
            self.final_state = state
            output = tf.reshape(outputs, [-1, self.args.rnn_size])

        if self.args.is_training:
            logits = tf.matmul(output, softmax_W) + softmax_b
            label = tf.reshape(self.Y, (-1,))
            loss = legacy_seq2seq.sequence_loss_by_example([logits], [self.Y], [tf.ones([tf.shape(logits)[0]])])
            self.nan = tf.reduce_sum(tf.to_float(tf.is_nan(loss)))
            mask_loss = loss * self.mask
            self.cost = tf.reduce_sum(mask_loss) / tf.reduce_sum(self.mask)
        else:
            self.prediction = logits = tf.matmul(output, softmax_W) + softmax_b
            self.hit_at_k, self.ndcg_at_k, self.num_target = self._metric_at_k()

        if not self.args.is_training:
            return

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_steps, self.args.decay, staircase=True)) 
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.args.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.args.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs 
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

   
    def _metric_at_k(self, k=20):
        prediction = self.prediction
        prediction_transposed = tf.transpose(prediction)
        labels = tf.reshape(self.Y, shape=(-1,))
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.args.n_items])
        ranks = tf.reduce_sum(tf.cast(prediction[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1

        ndcg = 1. / (log2(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k) # also known as Recall@k
        hit_at_k = tf.cast(hit_at_k, dtype=tf.float32)
        istarget = tf.reshape(self.mask, shape=(-1,))
        hit_at_k *= istarget
        ndcg_at_k = ndcg * istarget

        return (tf.reduce_sum(hit_at_k), tf.reduce_sum(ndcg_at_k), tf.reduce_sum(istarget))
def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
