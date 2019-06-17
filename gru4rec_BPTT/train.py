# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os, sys
import tensorflow as tf
import numpy as np
import argparse, random

from model import GRU4Rec
from utils import load_train, load_valid

unfold_max = 20
error_during_training = False

train_x, train_y, n_items = load_train(unfold_max)
valid_x, valid_y, _ = load_valid(unfold_max)

class Args():
    is_training = True
    layers = 1
    rnn_size = 100
    n_epochs = 10
    batch_size = 50
    keep_prob=1
    learning_rate = 0.001
    decay = 0.98
    decay_steps = 2*1e3
    sigma = 0.0001
    init_as_normal = False
    grad_cap = 0
    checkpoint_dir = 'save/{}'.format('lstm')
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1
    n_users = 1000
    init_from = None
    eval_point = 1*1e2

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='LSTM4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dr', default=0.98, type=float)
    parser.add_argument('--ds', default=50, type=int)
    parser.add_argument('--keep', default='1.0', type=float)
    parser.add_argument('--init_from', default=None, type=str)
    command_line = parser.parse_args()
    
    args.layers = command_line.layer
    args.batch_size = command_line.batch
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.decay = command_line.dr
    args.decay_steps = command_line.ds
    args.rnn_size = command_line.size
    args.keep_prob = command_line.keep
    args.checkpoint_dir += ('_p' + str(command_line.keep))
    args.checkpoint_dir += ('_rnn' + str(command_line.size))
    args.checkpoint_dir += ('_batch'+str(command_line.batch))
    args.checkpoint_dir += ('_lr'+str(command_line.lr))
    args.checkpoint_dir += ('_dr'+str(command_line.dr))
    args.checkpoint_dir += ('_ds'+str(command_line.ds))
    args.checkpoint_dir += ('_unfold'+str(unfold_max))
    args.init_from = command_line.init_from
    return args

def train(args):
    # Read train and test data. 
    global n_items, train_x, train_y
    args.n_items = n_items
    print('#Items: {}'.format(n_items))
    print('#Training sessions: {}'.format(len(train_x)))
    sys.stdout.flush()
    # set gpu configuations.
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = GRU4Rec(args)
        if args.init_from is not None:
            ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(sess, ckpt.model_checkpoint_path)
                print 'Restore model from :'.format(args.checkpoint_dir)
        else:
            sess.run(tf.global_variables_initializer())
            print 'Randomly initialize model'
        valid_losses = []
        best_step = -1
        best_epoch = -1
        best_loss = 100.0
        error_during_train = False
        num_batches = len(train_x) / args.batch_size 

        data = list(zip(train_x, train_y))
        random.shuffle(data)
        train_x, train_y = zip(*data)

        for epoch in xrange(args.n_epochs):
            epoch_cost = []
            for k in xrange(num_batches - 1):

                in_data = train_x[k*args.batch_size: (k+1)*args.batch_size]
                out_data = train_y[k*args.batch_size: (k+1)*args.batch_size]
                assert not np.any(np.isnan(in_data))
                assert not np.any(np.isnan(out_data))
                fetches = [model.nan, model.cost, model.global_step, model.lr, model.train_op]
                feed_dict = {model.X: in_data, model.Y: out_data}
                xnan, cost, step, lr, _ = sess.run(fetches, feed_dict)
                epoch_cost.append(cost)
                if np.isnan(cost):
                    print(str(epoch) + ':Nan error!')
                    print(xnan)
                    error_during_train = True
                    return
                if step == 1 or step % args.decay_steps == 0:
                    avgc = np.mean(epoch_cost)
                    print('Epoch {}\tProgress {}/{}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, k, num_batches, lr, avgc))
                if step % args.eval_point == 0:
                    valid_loss = eval_validation(model, sess)
                    valid_losses.append(valid_loss)
                    print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))
                    if valid_loss < best_loss:
                        best_epoch = epoch
                        best_step = step
                        best_loss = valid_losses[-1]
                        ckpt_path = os.path.join(args.checkpoint_dir, 'model.ckpt')
                        model.saver.save(sess, ckpt_path, global_step=step)
                        print("model saved to {}".format(ckpt_path))
                        sys.stdout.flush()

        print('Best evaluation loss appears in epoch {}, step {}. Lowest loss: {:.6f}'.format(best_epoch, best_step, best_loss))
        return

def eval_validation(model, sess):
    global valid_x, valid_y
    valid_batches = len(valid_x) / args.batch_size
    valid_loss = []
    for k in xrange(valid_batches):
        in_data = valid_x[k*args.batch_size: (k+1)*args.batch_size]
        out_data = valid_y[k*args.batch_size: (k+1)*args.batch_size]

        feed_dict = {model.X: in_data,
                    model.Y: out_data,
        }
        fetches = model.cost
        cost = sess.run(fetches, feed_dict)
        if np.isnan(cost):
            print('Evaluation loss Nan!')
            sys.exit(1)
        valid_loss.append(cost)
    return np.mean(valid_loss)

if __name__ == '__main__':
    args = parseArgs()
    if not os.path.exists('save'):
        os.mkdir('save')
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    print('rnn size: {}\tlayer: {}\tbatch: {}\tepoch: {}\tkeep: {}'.format(args.rnn_size, args.layers, args.batch_size, args.n_epochs, args.keep_prob))
    train(args)
