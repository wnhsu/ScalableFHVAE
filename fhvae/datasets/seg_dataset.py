import numpy as np
from seq_dataset import *
from collections import defaultdict

class Segment(object):
    def __init__(self, seq, start, end, lab, talab):
        self.seq = seq
        self.start = start
        self.end = end
        self.lab = lab
        self.talab = talab
    
    def __str__(self):
        return "(%s, %s, %s, %s, %s)" % (
                self.seq, self.start, self.end, self.lab, self.talab)

    def __repr__(self):
        return str(self)

def make_segs(seqs, lens, labs, talabs, seg_len, seg_shift, rand_seg):
    """
    Args:
        seqs(list): list of sequences
        lens(list): list of sequence lengths
        labs(list): list of sequence label lists
        talabs(list): list of sequence time-aligned label sequence lists
        seg_len(int):
        seg_shift(int):
        rand_seg(bool):
    """
    segs = []
    nsegs = []
    for seq, l, lab, talab in zip(seqs, lens, labs, talabs):
        nseg = (l - seg_len) // seg_shift + 1
        nsegs.append(nseg)
        if rand_seg:
            starts = np.random.choice(xrange(l - seg_len + 1), nseg)
        else:
            starts = np.arange(nseg) * seg_shift
        for start in starts:
            end = start + seg_len
            seg_talab = [s.center_lab(start, end) for s in talab]
            segs.append(Segment(seq, start, end, lab, seg_talab))
    return segs, nsegs

class SegmentDataset(object):
    def __init__(self, seq_d, seg_len=20, seg_shift=8, rand_seg=False):
        """
        Args:
            seq_d(SequenceDataset): SequenceDataset or its child class
            seg_len(int): segment length
            seg_shift(int): segment shift if seg_rand is False; otherwise
                randomly extract floor(seq_len/seg_shift) segments per sequence
            rand_seg(bool): if randomly extract segments or not
        """
        self.seq_d = seq_d
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.rand_seg = rand_seg
        
        self.seqlist = self.seq_d.seqlist
        self.feats = self.seq_d.feats
        self.lens = self.seq_d.lens
        self.labs_d = self.seq_d.labs_d
        self.talabseqs_d = self.seq_d.talabseqs_d
    
    def seq_iterator(self, bs, lab_names=[], talab_names=[], seqs=None,
            shuffle=False, rem=True, mapper=None):
        return self.seq_d.iterator(
                bs, lab_names, talab_names, seqs, shuffle, rem, mapper)

    def iterator(self, seg_bs, seg_shift=None, rand_seg=None, seg_shuffle=False, 
            seg_rem=True, seq_bs=-1, lab_names=[], talab_names=[], seqs=None, 
            seq_shuffle=False, seq_rem=True, seq_mapper=None):
        """
        Args:
            seg_bs(int): segment batch size
            seg_shift(int): use self.seg_shift if not set (None)
            rand_seg(bool): use self.rand_seg if not set (None)
            seg_shuffle(bool): shuffle segment list if True
            seg_rem(bool): yield remained segment batch if True
            seq_bs(int): -1 for loading all sequences. otherwise only
                blocked randomization for segments available
            lab_names(list): see SequenceDataset
            talab_names(list): see SequenceDataset
            seqs(list): see SequenceDataset
            seq_shuffle(bool): shuffle sequence list if True. this is 
                unnecessary if seq_bs == -1 and seg_shuffle == True
            seq_rem(bool): yield remained sequence batch if True
            seq_mapper(callable): see SequenceDataset
        """
        seqs = self.seqlist if seqs is None else seqs
        seq_bs = len(seqs) if seq_bs == -1 else seg_bs
        seg_shift = self.seg_shift if seg_shift is None else seg_shift
        rand_seg = self.rand_seg if rand_seg is None else rand_seg

        seq_iterator = self.seq_iterator(seq_bs, lab_names, talab_names, 
                seqs, seq_shuffle, seq_rem, seq_mapper)
        for seq_keys, seq_feats, seq_lens, seq_labs, seq_talabs in seq_iterator:
            segs, seq_nsegs = make_segs(seq_keys, seq_lens, seq_labs, seq_talabs, 
                    self.seg_len, seg_shift, rand_seg)
            if seg_shuffle:
                np.random.shuffle(segs)

            keys, feats, nsegs, labs, talabs = [], [], [], [], []
            seq2idx = dict([(seq, i) for i, seq in enumerate(seq_keys)])
            for seg in segs:
                idx = seq2idx[seg.seq]
                keys.append(seq_keys[idx])
                feats.append(seq_feats[idx][seg.start:seg.end]) 
                nsegs.append(seq_nsegs[idx])
                labs.append(seg.lab)
                talabs.append(seg.talab)
                if len(keys) == seg_bs:
                    yield keys, feats, nsegs, labs, talabs
                    keys, feats, nsegs, labs, talabs = [], [], [], [], []
            if seg_rem and bool(keys):
                yield keys, feats, nsegs, labs, talabs

    def lab2nseg(self, lab_name, seg_shift=None):
        lab2nseg = defaultdict(int)
        seg_shift = self.seg_shift if seg_shift is None else seg_shift
        for seq in self.seqlist:
            nseg = (self.lens[seq] - self.seg_len) // seg_shift + 1
            lab = self.labs_d[lab_name][seq]
            lab2nseg[lab] += nseg
        return lab2nseg
    
    def get_shape(self):
        seq_shape = self.seq_d.get_shape()
        return (self.seg_len,) + seq_shape[1:]

class KaldiSegmentDataset(SegmentDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], min_len=1,
            preload=False, mvn_path=None, seg_len=20, seg_shift=8, rand_seg=False):
        seq_d = KaldiDataset(feat_scp, len_scp, lab_specs, talab_specs,
                min_len, preload, mvn_path)
        super(KaldiSegmentDataset, self).__init__(
                seq_d, seg_len, seg_shift, rand_seg)
 
class NumpySegmentDataset(SegmentDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], min_len=1,
            preload=False, mvn_path=None, seg_len=20, seg_shift=8, rand_seg=False):
        seq_d = NumpyDataset(feat_scp, len_scp, lab_specs, talab_specs,
                min_len, preload, mvn_path)
        super(NumpySegmentDataset, self).__init__(
                seq_d, seg_len, seg_shift, rand_seg)
