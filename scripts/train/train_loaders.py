import sys
import numpy as np
from fhvae.datasets.seg_dataset import KaldiSegmentDataset, NumpySegmentDataset

def load_data(name, is_numpy):
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20
    seg_shift = 8
    Dataset = NumpySegmentDataset if is_numpy else KaldiSegmentDataset
    
    tr_dset = Dataset(
            "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path, 
            seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)
    dt_dset = Dataset(
            "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path,
            seg_len=seg_len, seg_shift=seg_len, rand_seg=False)

    return _load(tr_dset, dt_dset)

def _load(tr_dset, dt_dset):
    def _make_batch(seqs, feats, nsegs, seq2idx):
        x = feats
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        return x, y, n
    
    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    def tr_iterator(bs=256):
        seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        _iterator = tr_dset.iterator(bs, seg_shuffle=True, seg_rem=False)
        for seqs, feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx)
    
    def dt_iterator(bs=2048):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True)
        for seqs, feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx)

    return tr_nseqs, tr_shape, tr_iterator, dt_iterator
