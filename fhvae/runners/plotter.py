import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(m_l, y_l, title, path):
    """
    scatter plot for 2D matrix m
    
    Args:
        m_l(list): list of n-by-2 matrix
        y_l(list): list of len(m) labels
        path(str): path to save image
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    p_l = []
    for m, y in zip(m_l, y_l):
        p_l.append(plt.scatter(m[:, 0], m[:, 1], alpha=.8))
    ax.legend(p_l, y_l)
    ax.set_title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
def plot_x(x_l, seqs, path, sep=False, clim=(-2., 2.)):
    """
    plot spectrogram
    
    Args:
        x_l(list): list of n-by-T-by-F numpy.ndarray matrix of segment spectrograms
        seqs(list): list of sequence names
        path(str): path to save image
        sep(bool): add separators between segments if True
        clim(tuple): tuple of color limit (min_val, max_val)
    """
    fig = plt.figure(figsize=(16, 16))
    nrows = len(x_l)
    
    x_l = pad_x(x_l)
    for i in xrange(len(x_l)):
        x_2d = to_img(x_l[i], sep)
        ax = fig.add_subplot(nrows, 1, i + 1)
        im = ax.imshow(x_2d, interpolation="none", origin="lower")
        ax.set_title(seqs[i])
        im.set_clim(*clim)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def pad_x(x_l):
    """
    pad list of x to the same length

    Args:
        x_l(list): see plot_x
    Return:
        list of max(n)-by-T-by-F matrix of padded sequence spectrogram
    """
    min_val = np.min([np.min(x) for x in x_l])
    max_n = np.max([len(x) for x in x_l])
    pad = np.ones_like(x_l[0][0]) * min_val

    for i in xrange(len(x_l)):
        this_pad = np.tile(pad, (max_n - x_l[i].shape[0], 1, 1))
        x_l[i] = np.concatenate([x_l[i], this_pad], axis=0)
    return x_l

def to_img(x, sep):
    """
    convert a n-by-T-by-F segment spectrograms to a 2d image

    Args:
        x(np.ndarray): n-by-T-by-F segment spectrograms
        sep(bool): add separators between segments if True
    """
    n, T, F = x.shape
    if sep:
        min_val = np.min(x)
        sep = np.ones((n, 1, F)) * min_val
        x = np.concatenate([x, sep], axis=1)
        T += 1
    return x.reshape((n * T, F)).transpose()
