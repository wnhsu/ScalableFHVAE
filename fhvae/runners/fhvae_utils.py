import sys
import cPickle
import numpy as np
from collections import defaultdict
import tensorflow as tf
from ..models.fhvae_fn import map_mu2_z2_sum, z2_mu_fn

def load_prog(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        prog(list):
        epoch(int):
        global_step(int):
        passes(int):
        best_epoch(int):
        best_dt_lb(float):
    """
    def _print(msg):
        if not quiet:
            print msg

    with open(prog_pkl, "rb") as f:
        prog = cPickle.load(f)
        epoch, global_step, passes, best_epoch, best_dt_lb, _, _ = prog[-1]
        epoch, passes = epoch + 1, passes + 1

    dt_sum_names = ["lb", "log_px_z", "neg_kld_z1", "neg_kld_z2", "log_pmu2"]
    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %5s %7s %6s]" % ("epoch", "step", "pass", "best", "best_lb", "time")
    _print(msg)
    for p in prog:
        msg = "[%5s %7s %5s %5s %.2f %6d]" % tuple(p[:-1])
        msg += " " + " ".join(["%.2f" % p[-1][k] for k in dt_sum_names])
        _print(msg)
    return prog, epoch, global_step, passes, best_epoch, best_dt_lb

def get_best_step(prog, quiet=False):
    """
    retrieve the step of the best epoch
    """
    def _print(msg):
        if not quiet:
            print msg

    best_epoch = prog[-1][3]
    p = prog[best_epoch - 1]
    _print("\nBest Epoch Statistics:")
    _print("[%5s %7s %5s %5s %.2f %6d]" % tuple(p[:-1]))
    return p[1]

def restore_model(sess, saver, model_dir, step=None):
    """
    restore model parameters.
    Args:
        sess(tf.Session):
        saver(tf.train.Saver):
        model_dir(str):
        step(None/int): restore model at the given step if not None;
            otherwise restore the latest model
    """
    if step is None:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt is not None and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model_path = ckpt.model_checkpoint_path
        else:
            raise ValueError("model not found in %s" % model_dir)
    else:
        model_path = "%s/fhvae-%s" % (model_dir, step)
        if not tf.train.checkpoint_exists(model_path):
            raise ValueError("model %s does not exists" % model_path)
    print "reading parameters from %s" % model_path
    saver.restore(sess, model_path)

def _valid(sess, model, sum_vars, iterator):
    """
    compute averaged outputs of summary variables over a dataset
    Args:
        sess(tf.Session):
        model(FHVAE):
        sum_vars(list): list of model output variables
        iterator(Callable): dataset iterator function that 
            generate x(feat), y(seqidx), n(nseg)
    Return:
        sum_vals(list): list of averaged output values. same order as sum_vars
    """
    mu2_dict = _est_mu2_dict(sess, model, iterator)
    _print_mu2_stat(mu2_dict)
    sum_vals = [0. for _ in xrange(len(sum_vars))]
    tot_segs = 0.
    for x_val, y_val, n_val in iterator(bs=2048):
        mu2_val = _make_mu2(mu2_dict, y_val)
        feed_dict = {model.xin: x_val, model.xout: x_val, 
                model.n:n_val, model.mu2:mu2_val}
        out = sess.run(sum_vars, feed_dict)
        for j, val in enumerate(out):
            sum_vals[j] += (val * len(x_val))
        tot_segs += len(x_val)
    for i in xrange(len(sum_vars)):
        sum_vals[i] /= (tot_segs)
    return sum_vals

def _est_mu2_dict(sess, model, iterator):
    """
    estimate mu2 for sequences produced by iterator
    Args:
        sess(tf.Session):
        model(FHVAE):
        iterator(Callable):
    Return:
        mu2_dict(dict): sequence index to mu2 dict
    """
    nseg_table = defaultdict(float)
    z2_sum_table = defaultdict(float)
    for x_val, y_val, _ in iterator():
        z2 = z2_mu_fn(sess, model, x_val)
        for _y, _z2 in zip(y_val, z2):
            z2_sum_table[_y] += _z2
            nseg_table[_y] += 1
    mu2_dict = dict()
    for _y in nseg_table:
        mu2_dict[_y] = map_mu2_z2_sum(model, z2_sum_table[_y], nseg_table[_y])
    return mu2_dict

def _make_mu2(mu2_dict, y):
    """
    make mu2 input for a batch
    Args:
        mu2_dict(dict): sequence index to mu2 dict
        y(list): list of sequence index
    Return:
        (numpy.ndarray): (len(y))-by-(mu2_dim) matrix
    """
    return np.array([mu2_dict[_y] for _y in y])

def _print_mu2_stat(mu2_dict):
    norm_sum = 0.
    dim_norm_sum = 0.
    for y in sorted(mu2_dict.keys()):
        norm_sum += np.linalg.norm(mu2_dict[y])
        dim_norm_sum += np.abs(mu2_dict[y])
    avg_norm = norm_sum / len(mu2_dict)
    avg_dim_norm = dim_norm_sum / len(mu2_dict)
    print "avg. norm = %.2f, #mu2 = %s" % (avg_norm, len(mu2_dict))
    print "per dim: %s" % (" ".join(["%.2f" % v for v in avg_dim_norm]),)
