import os
import sys
import time
import cPickle
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
import tensorflow as tf
from ..models.fhvae_fn import z1_mu_fn, z2_mu_fn, x_mu_fn, x_logvar_fn, map_mu2_z2
from .fhvae_utils import restore_model, _valid
from .plotter import plot_x, scatter_plot

SESS_CONF = tf.ConfigProto(allow_soft_placement=True)
SESS_CONF.gpu_options.per_process_gpu_memory_fraction = 0.9

np.random.seed(123)

def test(exp_dir, step, model, iterator):
    """
    compute variational lower bound
    """
    xin, xout, y, n = model.xin, model.xout, model.y, model.n
    sum_names = ["lb", "log_px_z", "neg_kld_z1", "neg_kld_z2", "log_pmu2"]
    sum_vars = map(tf.reduce_mean, [model.__dict__[name] for name in sum_names])
    
    # util fn
    def _feed_dict(x_val, y_val, n_val):
        return {xin: x_val, xout:x_val, y:y_val, n:n_val}

    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        stime = time.time()
        restore_model(sess, saver, "%s/models" % exp_dir, step)
        print "restore model takes %.2f s" % (time.time() - stime)
    
        def _print_prog(sum_names, sum_vals):
            msg = " ".join(["%s=%.2f" % p for p in zip(sum_names, sum_vals)])
            print msg
            if np.isnan(sum_vals[0]):
                if_diverged = True
            sys.stdout.flush()
        
        print("#" * 40)
        vtime = time.time()
        sum_vals = _valid(sess, model, sum_vars, iterator)
        now = time.time()
        print("validation takes %.fs" % (now - vtime,))
        _print_prog(sum_names, sum_vals)
        print("#" * 40)

def visualize(exp_dir, step, model, iterator_by_seqs, seqs):
    """
    visualize reconstruction, factorization, sequence-translation
    """
    if len(seqs) > 10:
        print(">10 seqs. randomly select 10 seqs for visualization")
        seqs = sorted(list(np.random.choice(seqs, 10, replace=False)))
    
    try:
        os.makedirs("%s/img" % exp_dir)
    except OSError:
        pass

    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        stime = time.time()
        restore_model(sess, saver, "%s/models" % exp_dir, step)
        print "restore model takes %.2f s" % (time.time() - stime)
   
        # infer z1, z2, mu2
        z1_by_seq = defaultdict(list)
        z2_by_seq = defaultdict(list)
        mu2_by_seq = dict()
        xin_by_seq = defaultdict(list)
        xout_by_seq = defaultdict(list)
        xoutv_by_seq = defaultdict(list)
        for seq in seqs:
            for x, _, _ in iterator_by_seqs([seq]):
                z2 = z2_mu_fn(sess, model, x)
                z1 = z1_mu_fn(sess, model, x, z2)
                xout = x_mu_fn(sess, model, z1, z2)
                xoutv = x_logvar_fn(sess, model, z1, z2)
                z1_by_seq[seq].append(z1)
                z2_by_seq[seq].append(z2)
                xin_by_seq[seq].append(x)
                xout_by_seq[seq].append(xout)
                xoutv_by_seq[seq].append(xoutv)
            z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
            z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)
            xin_by_seq[seq] = np.concatenate(xin_by_seq[seq], axis=0)
            xout_by_seq[seq] = np.concatenate(xout_by_seq[seq], axis=0)
            xoutv_by_seq[seq] = np.concatenate(xoutv_by_seq[seq], axis=0)
            mu2_by_seq[seq] = map_mu2_z2(model, z2_by_seq[seq])

        seq_names = ["%02d_%s" % (i, seq) for i, seq in enumerate(seqs)]

        # visualize reconstruction
        print "visualizing reconstruction"
        plot_x([xin_by_seq[seq] for seq in seqs], seq_names, "%s/img/xin.png" % exp_dir)
        plot_x([xout_by_seq[seq] for seq in seqs], seq_names, "%s/img/xout.png" % exp_dir)
        plot_x([xoutv_by_seq[seq] for seq in seqs], seq_names, 
                "%s/img/xout_logvar.png" % exp_dir, clim=(None, None))
        
        # factorization: use the centered segment from each sequence
        print "visualizing factorization"
        cen_z1 = np.array([z1_by_seq[seq][len(z1_by_seq[seq]) / 2] for seq in seqs])
        cen_z2 = np.array([z2_by_seq[seq][len(z2_by_seq[seq]) / 2] for seq in seqs])
        xfac = []
        for z1 in cen_z1:
            z1 = np.tile(z1, (len(cen_z2), 1))
            xfac.append(x_mu_fn(sess, model, z1, cen_z2))
        plot_x(xfac, seq_names, "%s/img/xfac.png" % exp_dir, sep=True)

        # sequence translation
        print "visualizing sequence translation"
        xtra_by_seq = dict()
        for src_seq, src_seq_name in zip(seqs, seq_names):
            xtra_by_seq[src_seq] = dict()
            src_z1, src_z2 = z1_by_seq[src_seq], z2_by_seq[src_seq]
            for tar_seq in seqs:
                del_mu2 = mu2_by_seq[tar_seq] - mu2_by_seq[src_seq]
                xtra_by_seq[src_seq][tar_seq] = _seq_translate(
                        sess, model, src_z1, src_z2, del_mu2)
            plot_x([xtra_by_seq[src_seq][seq] for seq in seqs], seq_names, 
                    "%s/img/%s_tra.png" % (exp_dir, src_seq_name))

        # tsne z1 and z2
        print "t-SNE analysis on latent variables"
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)
        
        p = 30
        print "  perplexity = %s" % p
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne = _unflatten(tsne.fit_transform(z1), n)
        scatter_plot(z1_tsne, seq_names, "z1_tsne_%03d" % p, 
                "%s/img/z1_tsne_%03d.png" % (exp_dir, p))
        z2_tsne = _unflatten(tsne.fit_transform(z2), n)
        scatter_plot(z2_tsne, seq_names, "z2_tsne_%03d" % p, 
                "%s/img/z2_tsne_%03d.png" % (exp_dir, p))

def tsne_by_label(exp_dir, step, model, iterator_by_seqs, seqs, seq2lab_d):
    saver = tf.train.Saver()
    with tf.Session(config=SESS_CONF) as sess:
        restore_model(sess, saver, "%s/models" % exp_dir, step)
   
        # infer z1, z2
        z1_by_seq = defaultdict(list)
        z2_by_seq = defaultdict(list)
        for seq in seqs:
            for x, _, _ in iterator_by_seqs([seq]):
                z2 = z2_mu_fn(sess, model, x)
                z1 = z1_mu_fn(sess, model, x, z2)
                z1_by_seq[seq].append(z1)
                z2_by_seq[seq].append(z2)
            z1_by_seq[seq] = np.concatenate(z1_by_seq[seq], axis=0)
            z2_by_seq[seq] = np.concatenate(z2_by_seq[seq], axis=0)

        # tsne z1 and z2
        print "t-SNE analysis on latent variables by label"
        n = [len(z1_by_seq[seq]) for seq in seqs]
        z1 = np.concatenate([z1_by_seq[seq] for seq in seqs], axis=0)
        z2 = np.concatenate([z2_by_seq[seq] for seq in seqs], axis=0)
        
        p = 30
        tsne = TSNE(n_components=2, verbose=0, perplexity=p, n_iter=1000)
        z1_tsne_by_seq = dict(zip(seqs, _unflatten(tsne.fit_transform(z1), n)))
        for gen_fac, seq2lab in seq2lab_d.items():
            _labs, _z1 = _join(z1_tsne_by_seq, seq2lab)
            scatter_plot(_z1, _labs, gen_fac, 
                    "%s/img/tsne_by_label_z1_%s_%03d.png" % (exp_dir, gen_fac, p))
        
        z2_tsne_by_seq = dict(zip(seqs, _unflatten(tsne.fit_transform(z2), n)))
        for gen_fac, seq2lab in seq2lab_d.items():
            _labs, _z2 = _join(z2_tsne_by_seq, seq2lab)
            scatter_plot(_z2, _labs, gen_fac, 
                    "%s/img/tsne_by_label_z2_%s_%03d.png" % (exp_dir, gen_fac, p))

def _unflatten(l_flat, n_l):
    """
    unflatten a list
    """
    l = []
    offset = 0
    for n in n_l:
        l.append(l_flat[offset:offset+n])
        offset += n
    assert(offset == len(l_flat))
    return l

def _join(z_by_seqs, seq2lab):
    d = defaultdict(list)
    for seq, z in z_by_seqs.items():
        d[seq2lab[seq]].append(z)
    for lab in d:
        d[lab] = np.concatenate(d[lab], axis=0)
    return d.keys(), d.values()

def _seq_translate(sess, model, src_z1, src_z2, del_mu2):
    mod_z2 = src_z2 + del_mu2[np.newaxis, ...]
    return x_mu_fn(sess, model, src_z1, mod_z2)
