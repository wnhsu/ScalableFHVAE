import sys
import numpy as np
from fhvae.datasets.seg_dataset import KaldiSegmentDataset, NumpySegmentDataset

np.random.seed(123)

def load_data(name, set_name, is_numpy, seqlist_path):
    root = "./datasets/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    seg_len = 20
    Dataset = NumpySegmentDataset if is_numpy else KaldiSegmentDataset
    
    dt_dset = Dataset(
            "%s/%s/feats.scp" % (root, set_name), 
            "%s/%s/len.scp" % (root, set_name),
            min_len=seg_len, preload=False, mvn_path=mvn_path,
            seg_len=seg_len, seg_shift=seg_len, rand_seg=False)
    if seqlist_path is None:
        dt_seqs = np.random.choice(dt_dset.seqlist, 10)
        dt_seq2lab_d = None
    else:
        dt_seqs, dt_seq2lab_d = _load_seqlist(seqlist_path)
        assert(bool(dt_seqs))

    dt_iterator, dt_iterator_by_seqs, dt_seqs = _load(dt_dset, dt_seqs)
    return dt_iterator, dt_iterator_by_seqs, dt_seqs, dt_seq2lab_d

def _load(dt_dset, dt_seqs):
    def _make_batch(seqs, feats, nsegs, seq2idx):
        x = feats
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        return x, y, n
   
    def dt_iterator(bs=2048):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True)
        for seqs, feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx)

    def dt_iterator_by_seqs(s_seqs, bs=2048):
        seq2idx = dict([(seq, i) for i, seq in enumerate(s_seqs)])
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True, seqs=s_seqs)
        for seqs, feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, feats, nsegs, seq2idx)

    return dt_iterator, dt_iterator_by_seqs, dt_seqs

def _load_seqlist(seqlist_path):
    """
    header line is "#seq <gen_fac_1> <gen_fac_2> ..."
    each line is "<seq> <label_1> <label_2> ..."
    """
    seqs = []
    gen_facs = None
    seq2lab_l = None

    with open(seqlist_path) as f:
        for line in f:
            if line[0] == "#":
                gen_facs = line[1:].rstrip().split()[1:]
                seq2lab_l = [dict() for _ in gen_facs]
            else:
                toks = line.rstrip().split()
                seq = toks[0]
                labs = toks[1:]
                seqs.append(seq)
                if gen_facs is not None:
                    assert(len(seq2lab_l) == len(labs))
                    for seq2lab, lab in zip(seq2lab_l, labs):
                        seq2lab[seq] = lab

    if gen_facs is None:
        seq2lab_d = None
    else:
        seq2lab_d = dict(zip(gen_facs, seq2lab_l))

    return seqs, seq2lab_d
