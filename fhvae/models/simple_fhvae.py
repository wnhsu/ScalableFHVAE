import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import xavier_initializer
sce_logits = tf.nn.sparse_softmax_cross_entropy_with_logits

class SimpleFHVAE(object):
    def __init__(self, xin, xout, y, n, nmu2):
        # encoder/decoder arch
        self.z1_hus, self.z1_dim = [128, 128], 16
        self.z2_hus, self.z2_dim = [128, 128], 16
        self.x_hus = [128, 128]

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
                    self.xin, self.xout, self.y, self.nmu2, self.z1_hus, 
                    self.z1_dim, self.z2_hus, self.z2_dim, self.x_hus)

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

    def net(self, xin, xout, y, nmu2, z1_hus, z1_dim, z2_hus, z2_dim, x_hus):
        with tf.variable_scope("fhvae"):
            mu2_table, mu2 = mu2_lookup(y, z2_dim, nmu2)    
    
            z2_pre_out = z2_pre_encoder(xin, z2_hus)
            z2_mu, z2_logvar, z2_sample = gauss_layer(
                    z2_pre_out, z2_dim, scope="z2_enc_gauss")
            qz2_x = [z2_mu, z2_logvar]

            z1_pre_out = z1_pre_encoder(xin, z2_sample, z1_hus)
            z1_mu, z1_logvar, z1_sample = gauss_layer(
                    z1_pre_out, z1_dim, scope="z1_enc_gauss")
            qz1_x = [z1_mu, z1_logvar]
            
            x_pre_out = pre_decoder(z1_sample, z2_sample, x_hus)
            T, F = xout.get_shape().as_list()[1:]
            x_mu, x_logvar, x_sample = gauss_layer(
                    x_pre_out, T * F, scope="x_dec_gauss")
            x_mu = tf.reshape(x_mu, (-1, T, F))
            x_logvar = tf.reshape(x_logvar, (-1, T, F))
            x_sample = tf.reshape(x_sample, (-1, T, F))
            px_z = [x_mu, x_logvar]
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
        msg += "\n      FC hidden units: %s" % str(self.z1_hus)
        msg += "\n      latent dim: %s" % self.z1_dim
        msg += "\n    z2 encoder:"
        msg += "\n      FC hidden units: %s" % str(self.z2_hus)
        msg += "\n      latent dim: %s" % self.z2_dim
        msg += "\n    mu2 table size: %s" % self.nmu2
        msg += "\n    x decoder:"
        msg += "\n      FC hidden units: %s" % str(self.x_hus)
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

def z1_pre_encoder(x, z2, hus=[1024, 1024]):
    """
    Pre-stochastic layer encoder for z1 (latent segment variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        z2(tf.Tensor): tensor of shape (bs, D1)
        hus(list): list of numbers of FC layer hidden units
    Return:
        out(tf.Tensor): last FC layer output
    """
    with tf.variable_scope("z1_pre_enc"):
        T, F = x.get_shape().as_list()[1:]
        x = tf.reshape(x, (-1, T * F))
        out = tf.concat([x, z2], axis=-1)
        for i, hu in enumerate(hus):
            out = fully_connected(out, hu, activation_fn=tf.nn.relu, scope="fc%s" % i)
    return out

def z2_pre_encoder(x, hus=[1024, 1024]):
    """
    Pre-stochastic layer encoder for z2 (latent sequence variable)
    Args:
        x(tf.Tensor): tensor of shape (bs, T, F)
        hus(list): list of numbers of LSTM layer hidden units
    Return:
        out(tf.Tensor): concatenation of hidden states of all LSTM layers
    """
    with tf.variable_scope("z2_pre_enc"):
        T, F = x.get_shape().as_list()[1:]
        out = tf.reshape(x, (-1, T * F))
        for i, hu in enumerate(hus):
            out = fully_connected(out, hu, activation_fn=tf.nn.relu, scope="fc%s" % i)
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

def pre_decoder(z1, z2, hus=[1024, 1024]):
    """
    Pre-stochastic layer decoder
    Args:
        z1(tf.Tensor)
        z2(tf.Tensor)
        x(tf.Tensor): tensor of shape (bs, T, F). only shape is used
        hus(list)
    """
    with tf.variable_scope("dec"):
        out = tf.concat([z1, z2], axis=-1)
        for i, hu in enumerate(hus):
            out = fully_connected(out, hu, activation_fn=tf.nn.relu, scope="fc%s" % i)
    return out

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
