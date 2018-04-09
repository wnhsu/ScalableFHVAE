import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import \
        BasicLSTMCell, MultiRNNCell
sce_logits = tf.nn.sparse_softmax_cross_entropy_with_logits

class FHVAE(object):
    def __init__(self, xin, xout, y, n, nmu2):
        # encoder/decoder arch
        self.z1_rhus, self.z1_dim = [256, 256], 32
        self.z2_rhus, self.z2_dim = [256, 256], 32
        self.x_rhus = [256, 256]

        # observed vars
        self.xin = xin
        self.xout = xout
        self.y = y
        self.n = n
        self.nmu2 = nmu2
        
        # latent vars
        self.mu2_table, self.mu2, self.qz2_x, self.z2_sample, self.qz1_x, \
                self.z1_sample, self.px_z, self.x_sample = \
                self.net(
                    self.xin, self.xout, self.y, self.nmu2, self.z1_rhus, 
                    self.z1_dim, self.z2_rhus, self.z2_dim, self.x_rhus)

        # priors
        self.pz1 = [0., np.log(1.0 ** 2).astype(np.float32)]
        self.pz2 = [self.mu2, np.log(0.5 ** 2).astype(np.float32)]
        self.pmu2 = [0., np.log(1.0 ** 2).astype(np.float32)]

        # variational lower bound
        self.log_pmu2 = tf.reduce_sum(
                log_gauss(self.mu2, self.pmu2[0], self.pmu2[1]), axis=1)
        self.neg_kld_z2 = -1 * tf.reduce_sum(
                kld(self.qz2_x[0], self.qz2_x[1], self.pz2[0], self.pz2[1]), axis=1)
        self.neg_kld_z1 = -1 * tf.reduce_sum(
                kld(self.qz1_x[0], self.qz1_x[1], self.pz1[0], self.pz1[1]), axis=1)
        self.log_px_z = tf.reduce_sum(
                log_gauss(xout, self.px_z[0], self.px_z[1]), axis=(1, 2))
        self.lb = self.log_px_z + self.neg_kld_z1 + self.neg_kld_z2 + self.log_pmu2 / n

        # discriminative loss
        logits = tf.expand_dims(self.qz2_x[0], 1) - tf.expand_dims(self.mu2_table, 0)
        logits = -1 * tf.pow(logits, 2) / (2 * tf.exp(self.pz2[1]))
        logits = tf.reduce_sum(logits, axis=-1)
        self.log_qy = -sce_logits(labels=y, logits=logits)

        # collect params
        self.params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    def net(self, xin, xout, y, nmu2, z1_rhus, z1_dim, z2_rhus, z2_dim, x_rhus):
        with tf.variable_scope("fhvae"):
            mu2_table, mu2 = mu2_lookup(y, z2_dim, nmu2)    
    
            z2_pre_out = z2_pre_encoder(xin, z2_rhus)
            z2_mu, z2_logvar, z2_sample = gauss_layer(
                    z2_pre_out, z2_dim, scope="z2_enc_gauss")
            qz2_x = [z2_mu, z2_logvar]

            z1_pre_out = z1_pre_encoder(xin, z2_sample, z1_rhus)
            z1_mu, z1_logvar, z1_sample = gauss_layer(
                    z1_pre_out, z1_dim, scope="z1_enc_gauss")
            qz1_x = [z1_mu, z1_logvar]
            
            x_pre_out, px_z, x_sample = decoder(
                    z1_sample, z2_sample, xout, x_rhus)
        return mu2_table, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample
 
    def __str__(self):
        msg = ""
        msg += "\nFactorized Hierarchical Variational Autoencoder:"
        msg += "\n  Priors (mean/logvar):"
        msg += "\n    pz1: %s" % str(self.pz1)
        msg += "\n    pz2: %s" % str(self.pz2)
        msg += "\n    pmu2: %s" % str(self.pmu2)
        msg += "\n  Observed Variables:"
        msg += "\n    xin: %s" % self.xin
        msg += "\n    xout: %s" % self.xout
        msg += "\n    y: %s" % self.y
        msg += "\n    n: %s" % self.n
        msg += "\n  Encoder/Decoder Architectures:"
        msg += "\n    z1 encoder:"
        msg += "\n      LSTM hidden units: %s" % str(self.z1_rhus)
        msg += "\n      latent dim: %s" % self.z1_dim
        msg += "\n    z2 encoder:"
        msg += "\n      LSTM hidden units: %s" % str(self.z2_rhus)
        msg += "\n      latent dim: %s" % self.z2_dim
        msg += "\n    mu2 table size: %s" % self.nmu2
        msg += "\n    x decoder:"
        msg += "\n      LSTM hidden units: %s" % str(self.x_rhus)
        msg += "\n  Outputs:"
        msg += "\n    qz1_x: %s" % str(self.qz1_x)
        msg += "\n    qz2_x: %s" % str(self.qz2_x)
        msg += "\n    mu2: %s" % str(self.mu2)
        msg += "\n    px_z: %s" % str(self.px_z)
        msg += "\n    z1_sample: %s" % str(self.z1_sample)
        msg += "\n    z2_sample: %s" % str(self.z2_sample)
        msg += "\n    x_sample: %s" % str(self.x_sample)
        msg += "\n  Losses:"
        msg += "\n    lb: %s" % str(self.lb)
        msg += "\n    log_px_z: %s" % str(self.log_px_z)
        msg += "\n    neg_kld_z1: %s" % str(self.neg_kld_z1)
        msg += "\n    neg_kld_z2: %s" % str(self.neg_kld_z2)
        msg += "\n    log_pmu2: %s" % str(self.log_pmu2)
        msg += "\n    log_qy: %s" % str(self.log_qy)
        msg += "\n  Parameters:"
        for param in self.params:
            msg += "\n    %s, %s" % (param.name, param.get_shape())
        return msg
       
def mu2_lookup(y, z2_dim, nmu2, init_std=1.0):
    """
    mu2 posterior mean lookup table
    Args:
        y(tf.Tensor): int tensor of shape (bs,). index for mu2_table
        z2_dim(int): z2 dimension
        nmu2(int): lookup table size
    """
    with tf.variable_scope("mu2"):
        init_val = tf.random_normal([nmu2, z2_dim], stddev=init_std)
        mu2_table = tf.get_variable("mu2_table", trainable=True, initializer=init_val)
        mu2 = tf.gather(mu2_table, y, name="mu2")
    return mu2_table, mu2

def z1_pre_encoder(x, z2, rhus=[256, 256]):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        z2(tf.Tensor): tensor of shape (bs, D1)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    bs, T = tf.shape(x)[0], tf.shape(x)[1]
    z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
    x_z2 = tf.concat([x, z2], axis=-1)

    cell = MultiRNNCell([BasicLSTMCell(rhu) for rhu in rhus])
    init_state = cell.zero_state(bs, x.dtype)
    name = "z1_enc_lstm_%s" % ("_".join(map(str, rhus)),)
    _, final_state = dynamic_rnn(cell, x_z2, dtype=x.dtype,
            initial_state=init_state, time_major=False, scope=name)
    
    out = [l_final_state.h for l_final_state in final_state]
    out = tf.concat(out, axis=-1)
    return out

def z2_pre_encoder(x, rhus=[256, 256]):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        rhus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    bs = tf.shape(x)[0]
    
    cell = MultiRNNCell([BasicLSTMCell(rhu) for rhu in rhus])
    init_state = cell.zero_state(bs, x.dtype)
    name = "z2_enc_lstm_%s" % ("_".join(map(str, rhus)),)
    _, final_state = dynamic_rnn(cell, x, dtype=x.dtype,
            initial_state=init_state, time_major=False, scope=name)
    
    out = [l_final_state.h for l_final_state in final_state]
    out = tf.concat(out, axis=-1)
    return out

def gauss_layer(inp, dim, mu_nl=None, logvar_nl=None, scope=None):
    """
    Gaussian layer
    Args:
        inp(tf.Tensor): input to Gaussian layer
        dim(int): dimension of output latent variables
        mu_nl(callable): nonlinearity for Gaussian mean
        logvar_nl(callable): nonlinearity for Gaussian log variance
        scope(str/VariableScope): tensorflow variable scope
    """
    with tf.variable_scope(scope, "gauss") as sc:
        mu = fully_connected(inp, dim, activation_fn=mu_nl,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="mu")
        logvar = fully_connected(inp, dim, activation_fn=logvar_nl,
                weights_initializer=xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                scope="logvar")
        eps = tf.random_normal(tf.shape(logvar), name='eps')
        sample = mu + tf.exp(0.5 * logvar) * eps
    return mu, logvar, sample

def decoder(z1, z2, x, rhus=[256, 256], x_mu_nl=None, x_logvar_nl=None):
    """
    decoder
    Args:
        z1(tf.Tensor)
        z2(tf.Tensor)
        x(tf.Tensor): tensor of shape (bs, T, F). only shape is used
        rhus(list)
    """
    bs = tf.shape(x)[0]
    z1_z2 = tf.concat([z1, z2], axis=-1)
    
    cell = MultiRNNCell([BasicLSTMCell(rhu) for rhu in rhus])
    state_t = cell.zero_state(bs, x.dtype)
    name = "dec_lstm_%s_step" % ("_".join(map(str, rhus)),)
    def cell_step(inp, prev_state):
        return cell(inp, prev_state, scope=name)
    
    gdim = x.get_shape().as_list()[2]
    gname = "dec_gauss_step"
    def glayer_step(inp):
        return gauss_layer(inp, gdim, x_mu_nl, x_logvar_nl, gname)

    out, x_mu, x_logvar, x_sample = [], [], [], []
    for t in xrange(x.get_shape().as_list()[1]):
        if t > 0:
            tf.get_variable_scope().reuse_variables()

        out_t, state_t, x_mu_t, x_logvar_t, x_sample_t = decoder_step(
                z1_z2, state_t, cell_step, glayer_step)
        out.append(out_t)
        x_mu.append(x_mu_t)
        x_logvar.append(x_logvar_t)
        x_sample.append(x_sample_t)

    out = tf.stack(out, axis=1, name="dec_pre_out")
    x_mu = tf.stack(x_mu, axis=1, name="dec_x_mu")
    x_logvar = tf.stack(x_logvar, axis=1, name="dec_x_logvar")
    x_sample = tf.stack(x_sample, axis=1, name="dec_x_sample")
    px_z = [x_mu, x_logvar]
    return out, px_z, x_sample

def decoder_step(inp_t, prev_state, cell_step, glayer_step):
    out_t, state_t = cell_step(inp_t, prev_state)
    x_mu_t, x_logvar_t, x_sample_t = glayer_step(out_t)
    return out_t, state_t, x_mu_t, x_logvar_t, x_sample_t

def log_gauss(x, mu=0., logvar=0.):
    """
    compute log N(x; mu, exp(logvar))
    """
    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(x - mu) / tf.exp(logvar))

def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    return -0.5 * (1 + p_logvar - q_logvar - \
            (tf.square(p_mu - q_mu) + tf.exp(p_logvar)) / tf.exp(q_logvar))
