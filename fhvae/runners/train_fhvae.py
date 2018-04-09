import os
import sys
import time
import cPickle
import numpy as np
import tensorflow as tf
from .fhvae_utils import load_prog, _valid

SESS_CONF = tf.ConfigProto(allow_soft_placement=True)
SESS_CONF.gpu_options.per_process_gpu_memory_fraction = 0.9

def train(exp_dir, model, args, train_conf, tr_iterator, dt_iterator):
    xin, xout, y, n = model.xin, model.xout, model.y, model.n
    n_epochs, n_patience, n_steps_per_epoch, n_print_steps, alpha_dis = train_conf
    adam_params = {"learning_rate":0.001, "beta1": 0.95, "beta2": 0.999}
   
    # train objective
    loss = -1 * tf.reduce_mean(model.lb + alpha_dis * model.log_qy)
    
    global_step_var = tf.Variable(0, trainable=False, name="global_step")
    opt = tf.train.AdamOptimizer(**adam_params)
    params = tf.trainable_variables()
    
    grads = opt.compute_gradients(loss, params)
    apply_grad_op = opt.apply_gradients(grads, global_step=global_step_var)
    
    # summary stats
    tr_sum_names = ["lb", "log_px_z", "neg_kld_z1", "neg_kld_z2", "log_pmu2", "log_qy"]
    tr_sum_vars = map(tf.reduce_mean, [model.__dict__[name] for name in tr_sum_names])
    dt_sum_names = ["lb", "log_px_z", "neg_kld_z1", "neg_kld_z2", "log_pmu2"]
    dt_sum_vars = map(tf.reduce_mean, [model.__dict__[name] for name in dt_sum_names])
 
    # set exp
    best_epoch, best_dt_lb = 0, -np.inf
    init_step, global_step, epoch, passes = 0, 0, 1, 1
    stime, ttime, etime, ptime = time.time(), time.time(), time.time(), time.time()
    prog = []

    # util fn
    def _check_terminate(epoch, best_epoch, n_patience, n_epochs):
        if (epoch - 1) - best_epoch > n_patience:
            return True
        if epoch > n_epochs:
            return True
        return False

    def _feed_dict(x_val, y_val, n_val):
        return {xin: x_val, xout:x_val, y:y_val, n:n_val}

    def _save_prog(dt_sum_vals):
        prog.append([epoch, global_step, passes, best_epoch, best_dt_lb, 
                time.time() - ttime, dict(zip(dt_sum_names, dt_sum_vals))])
        with open("%s/prog.pkl" % exp_dir, "wb") as f:
            cPickle.dump(prog, f)

    # create/load exp
    try:
        print "\nexp_dir: %s" % exp_dir
        os.makedirs("%s/models" % exp_dir)
    except OSError:
        prog_pkl = "%s/prog.pkl" % exp_dir
        prog, epoch, global_step, passes, best_epoch, best_dt_lb = load_prog(prog_pkl)
        if _check_terminate(epoch, best_epoch, n_patience, n_epochs):
            print "training was finished..."
            return
        print "\nStarting from:"
        print "  epoch = %s" % epoch
        print "  global_step = %s" % global_step
        print "  passes = %s" % passes
        print "  best_epoch = %s" % best_epoch
        print "  best_dt_lb = %.2f" % best_dt_lb

    with open("%s/args.pkl" % exp_dir, "wb") as f:
        cPickle.dump(args, f)
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(config=SESS_CONF) as sess:
        stime = time.time()
        if global_step == 0:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "%s/models/fhvae-%s" % (exp_dir, global_step))
        
        print "init or restore model takes %.2f s" % (time.time() - stime)
        print "current #steps=%s, #epochs=%s" % (global_step, epoch)
        print "start training..."
        ttime, etime, ptime = time.time(), time.time(), time.time()
    
        def _print_prog(sum_names, sum_vals):
            if_diverged = False
            now = time.time()
            msg = "[e%d/s%d/p%d]:" % (epoch, global_step, passes)
            msg += " time=%.f/%.f(s) " % (now - ptime, now - ttime)
            msg += " ".join(["%s=%.2f" % p for p in zip(sum_names, sum_vals)])
            print msg
            if np.isnan(sum_vals[0]):
                if_diverged = True
            sys.stdout.flush()
            return if_diverged
        
        def _valid_step():
            # assume dt_sum_vars[0] is variational lower bound
            is_best = False
            print("#" * 40)
            vtime = time.time()
            dt_sum_vals = _valid(sess, model, dt_sum_vars, dt_iterator)
            now = time.time()
            print("Finished epoch %s (%.fs/%.fs); validation takes %.fs" % (
                    epoch, now - ttime, now - etime, now - vtime))
            _print_prog(dt_sum_names, dt_sum_vals)
            print("#" * 40)
            if dt_sum_vals[0] > best_dt_lb:
                is_best = True
            return is_best, dt_sum_vals

        while True:
            for x_val, y_val, n_val in tr_iterator():
                feed_dict = _feed_dict(x_val, y_val, n_val)
                global_step, _ = sess.run([global_step_var, apply_grad_op], feed_dict)
    
                if global_step % n_print_steps == 0 and global_step != init_step:
                    feed_dict = _feed_dict(x_val, y_val, n_val)
                    tr_sum_vals = sess.run(tr_sum_vars, feed_dict)
                    is_diverged = _print_prog(tr_sum_names, tr_sum_vals)
                    if is_diverged:
                        print "training diverged..."
                        return
                    ptime = time.time()
    
                if global_step % n_steps_per_epoch == 0 and global_step != init_step:
                    is_best, dt_sum_vals = _valid_step()
                    if is_best:
                        best_epoch, best_dt_lb = epoch, dt_sum_vals[0]
                    saver.save(sess, "%s/models/fhvae" % exp_dir, global_step=global_step)
                    _save_prog(dt_sum_vals)
                    epoch += 1
                    if _check_terminate(epoch, best_epoch, n_patience, n_epochs):
                        print "training finished..."
                        return
                    etime = time.time()
            passes += 1
