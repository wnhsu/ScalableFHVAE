import numpy as np

def z2_mu_fn(sess, model, x):
    return sess.run(model.qz2_x[0], {model.xin: x})

def z1_mu_fn(sess, model, x, z2=None):
    if z2 is None:
        z2 = z2_mu_fn(sess, model, x)
    return sess.run(model.qz1_x[0], {model.xin: x, model.z2_sample: z2})

def x_mu_fn(sess, model, z1, z2):
    _, T, F = model.xout.get_shape().as_list()
    xout = np.zeros((len(z1), T, F))
    return sess.run(model.px_z[0], 
            {model.z1_sample: z1, model.z2_sample: z2, model.xout: xout})

def x_logvar_fn(sess, model, z1, z2):
    _, T, F = model.xout.get_shape().as_list()
    xout = np.zeros((len(z1), T, F))
    return sess.run(model.px_z[1], 
            {model.z1_sample: z1, model.z2_sample: z2, model.xout: xout})

def sample_z1_z2(sess, model, x):
    return sess.run([model.z1_sample, model.z2_sample], {model.xin: x})

def map_mu2_z2(model, z2):
    """
    approximated MAP estimation of mu2 given z2 for all segments
    Args:
        z2(numpy.ndarray): n-by-(z2_dim) float matrix of z2 for one sequence
            that has n segments
    """
    z2_sum = np.sum(z2, axis=0)
    n = len(z2)
    return map_mu2_z2_sum(model, z2_sum, n)

def map_mu2_z2_sum(model, z2_sum, n):
    """
    approximated MAP estimation of mu2 given sum of z2 over all segments
    Args:
        z2_sum(numpy.ndarray): matrix of the sample shape as z2. summed z2 over
            all segments within one sequence
        n(int): number of segments in the sequence
    """
    r = np.exp(model.pz2[1]) / np.exp(model.pmu2[1])
    return z2_sum / (n + r)

def update_mu2_table(sess, model, mu2_table):
    """
    update the mu2 table in model. used for training with hierarchical sampling
    Args:
        mu2(numpy.ndarray): model.nmu2-by-(z2_dim) matrix. new mu2_table to update
    """
    sess.run(model.mu2_table.assign(mu2_table), {})
