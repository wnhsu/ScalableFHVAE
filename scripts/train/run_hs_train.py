import os
import sys
import time
import argparse
import tensorflow as tf
from hs_train_loaders import load_data
from fhvae.models import load_model
from fhvae.runners.hs_train_fhvae import hs_train

print "I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime())

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="timit",
        help="dataset to use")
parser.add_argument("--is_numpy", action="store_true", dest="is_numpy",
        help="dataset format; kaldi by default")
parser.add_argument("--model", type=str, default="fhvae",
        help="model architecture; {fhvae|simple_fhvae}")
parser.add_argument("--alpha_dis", type=float, default=10.,
        help="discriminative objective weight")
parser.add_argument("--nmu2", type=int, default=5000,
        help="number of sequences for hierarchical sampling")
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_patience", type=int, default=10,
        help="number of maximum consecutive non-improving epochs")
parser.add_argument("--n_steps_per_epoch", type=int, default=5000,
        help="number of training steps per epoch")
parser.add_argument("--n_print_steps", type=int, default=200,
        help="number of steps to print statistics")
args = parser.parse_args()
print args

tr_nseqs, tr_shape, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator = \
        load_data(args.dataset, args.is_numpy)
FHVAE = load_model(args.model)

exp_root = "exp/%s" % args.dataset

xin = tf.placeholder(tf.float32, shape=(None,)+tr_shape, name="xin")
xout = tf.placeholder(tf.float32, shape=(None,)+tr_shape, name="xout")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
n = tf.placeholder(tf.float32, shape=(None,), name="n")
model = FHVAE(xin, xout, y, n, args.nmu2)
print(model)

# keep necessary information in args for restoring model
args.tr_shape = tr_shape

train_conf = [args.n_epochs, args.n_patience, args.n_steps_per_epoch,
        args.n_print_steps, args.alpha_dis, args.nmu2]
exp_dir = "%s/%s_hs_e%s_s%s_p%s_a%s_n%s" % (exp_root, args.model, args.n_epochs, 
        args.n_steps_per_epoch, args.n_patience, args.alpha_dis, args.nmu2)

hs_train(exp_dir, model, args, train_conf, sample_tr_seqs, tr_iterator_by_seqs, dt_iterator)
