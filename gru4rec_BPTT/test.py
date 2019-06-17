# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import sys
import numpy as np
import argparse
import tensorflow as tf

from model import GRU4Rec
from utils import load_test

unfold_max = 20
cut_off = 20

test_x, test_y, n_items = load_test(unfold_max)

class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 10
    batch_size = 50
    keep_prob = 1
    learning_rate = 0.002
    decay = 0.98
    decay_steps = 1e3*5
    sigma = 0.0005
    init_as_normal = False
    grad_cap = 0
    test_model = 9
    checkpoint_dir = 'save/{}'.format('lstm')
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dr', default=0.98, type=float)
    parser.add_argument('--ds', default=50, type=int)
    parser.add_argument('--keep', default='1.0', type=float)
    command_line = parser.parse_args()
    
    args.layers = command_line.layer
    args.batch_size = command_line.batch
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.rnn_size = command_line.size
    args.keep_prob = command_line.keep
    args.decay = command_line.dr
    args.decay_steps = command_line.ds
    args.checkpoint_dir += ('_p' + str(command_line.keep))
    args.checkpoint_dir += ('_rnn' + str(command_line.size))
    args.checkpoint_dir += ('_batch'+str(command_line.batch))
    args.checkpoint_dir += ('_lr'+str(command_line.lr))
    args.checkpoint_dir += ('_dr'+str(command_line.dr))
    args.checkpoint_dir += ('_ds'+str(command_line.ds))
    args.checkpoint_dir += ('_unfold'+str(unfold_max))
    return args

def evaluate(args):
    '''
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    '''
    args.n_items = n_items
    evalutation_point_count = 0
    mrr, recall, ndcg20, ndcg = 0.0, 0.0, 0.0, 0.0
    np.random.seed(42)
    
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    model = GRU4Rec(args)
    with tf.Session(config=gpu_config) as sess:
        #tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore model from {} successfully!'.format(args.checkpoint_dir))
        else:
            print('Restore model from {} failed!'.format(args.checkpoint_dir))
            return
        for i in xrange(len(test_x)):
            in_idx = np.asarray(test_x[i])
            out_idx = test_y[i]
            preds = model.predict_session(sess, in_idx)
            ranks = (preds > np.diag(preds[out_idx])).sum(axis=0) + 1
            #ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            rank_ok = ranks < cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            ndcg20 += (1.0 / np.log2(1.0+ranks[rank_ok])).sum()
            ndcg += (1.0 / np.log2(1.0+ranks)).sum()
            assert len(out_idx) == len(ranks)
            evalutation_point_count += len(ranks)
    return recall/evalutation_point_count, mrr/evalutation_point_count, ndcg20 / evalutation_point_count, ndcg / evalutation_point_count

if __name__ == '__main__':
    args = parseArgs()
    res = evaluate(args)
    print('lr: {}\tbatch_size: {}\tdecay_steps:{}\tdecay_rate:{}\tkeep_prob:{}\tdim: {}\tlayer: {}'.format(args.learning_rate, args.batch_size, args.decay_steps, args.decay, args.keep_prob, args.rnn_size, args.layers))
    print('Recall@20: {}\tMRR@20: {}\tNDCG@20: {}\tNDCG: {}'.format(res[0], res[1], res[2], res[3]))
    sys.stdout.flush()
