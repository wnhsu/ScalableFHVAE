import os
import numpy as np
import bisect
import cPickle
import librosa
from collections import OrderedDict
from kaldi_io import SequentialBaseFloatMatrixReader as SBFMReader
from kaldi_io import RandomAccessBaseFloatMatrixReader as RABFMReader
from .audio_utils import *

def scp2dict(path, dtype=str, seqlist=None):
    with open(path) as f:
        l = [line.rstrip().split(None, 1) for line in f]
    d = OrderedDict([(k, dtype(v)) for k, v in l])
    if seqlist is not None:
        d = subset_d(d, seqlist)
    return d

def load_lab(spec, seqlist=None):
    name, nclass, path = spec
    seq2lab = scp2dict(path, int, seqlist)
    return name, nclass, seq2lab

def subset_d(d, l):
    """
    retain keys in l. raise KeyError if some key is missed
    """
    new_d = OrderedDict()
    for k in l:
        new_d[k] = d[k]
    return new_d

def load_talab(spec, seqlist=None):
    name, nclass, path = spec
    with open(path) as f:
        toks_l = [line.rstrip().split() for line in f]
    assert(len(toks_l) > 0 and len(toks_l[0]) == 1)
    seq2talabseq = OrderedDict()
    seq = toks_l[0][0]
    talabs = []
    for toks in toks_l[1:]:
        if len(toks) == 1:
            seq2talabseq[seq] = TimeAlignedLabelSeq(talabs)
            seq = toks[0]
            talabs = []
        elif len(toks) == 3:
            talab = TimeAlignedLabel(int(toks[2]), int(toks[0]), int(toks[1]))
            talabs.append(talab)
        else:
            raise ValueError("invalid line %s" % str(toks))
    seq2talabseq[seq] = TimeAlignedLabelSeq(talabs)
    return name, nclass, seq2talabseq

class TimeAlignedLabel(object):
    """
    time-aligned label
    """
    def __init__(self, lab, start, stop):
        assert(start >= 0)
        self.lab = lab
        self.start = start
        self.stop = stop

    def __str__(self):
        return "(lab=%s, start=%s, stop=%s)" % (self.lab, self.start, self.stop)

    def __repr__(self):
        return str(self)

    @property
    def center(self):
        return (self.start + self.stop) / 2
    
    def __len__(self):
        return self.stop - self.start

    def centered_talab(self, slice_len):
        start = self.center - slice_len / 2
        stop = self.center + slice_len / 2
        return TimeAlignedLabel(self, self.lab, start, stop)

class TimeAlignedLabelSeq(object):
    """
    time-aligned labels for one sequence
    """
    def __init__(self, talabs, noov=True, nosp=False):
        """
        talabs(list): list of TimeAlignedLabel
        noov(bool): check no overlapping between TimeAlignedLabels
        nosp(bool): check no spacing between TimeAlignedLabels
        """
        talabs = sorted(talabs, key=lambda x: x.start)
        if noov and nosp:
            assert(talabs[0].start == 0)
            for i in xrange(len(talabs) - 1):
                if talabs[i].stop != talabs[i+1].start:
                    raise ValueError(talabs[i], talabs[i+1])
        elif noov:
            for i in xrange(len(talabs) - 1):
                if talabs[i].stop > talabs[i+1].start:
                    raise ValueError(talabs[i], talabs[i+1])
        elif nosp:
            assert(talabs[0].start == 0)
            for i in xrange(len(talabs) - 1):
                if talabs[i].stop < talabs[i+1].start:
                    raise ValueError(talabs[i], talabs[i+1])
        
        self.talabs = talabs
        self.noov = noov
        self.nosp = nosp
        self.max_stop = max([l.stop for l in talabs])

    def __str__(self):
        return "\n".join([str(l) for l in self.talabs])

    def __len__(self):
        return len(self.talabs)

    def to_seq(self):
        return [l.lab for l in self.talabs]

    def center_lab(self, start=-1, stop=-1, strict=False):
        """
        return the centered label in a sub-sequence
        Args:
            start(int)
            stop(int)
            strict(bool): raise error if center is not defined
        """
        if not self.noov:
            raise ValueError("center() only available in noov mode")

        start = 0 if start == -1 else start
        stop = self.max_stop if stop == -1 else stop
        center = (start + stop) / 2
        idx_l = bisect.bisect_right([l.start for l in self.talabs], center) - 1
        idx_r = bisect.bisect_right([l.stop for l in self.talabs], center)
        if not strict:
            return self.talabs[idx_l].lab
        elif idx_r != idx_l and strict:
            msg = "spacing detected at %s; " % center
            msg += "neigbors: %s, %s" % (self.talabs[idx_l], self.talabs[idx_r])
            raise ValueError(msg)
    
    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            self._lablist = sorted(np.unique([l.lab for l in self.talabs]))
        return self._lablist

class TimeAlignedLabelSeqs(object):
    """
    time-aligned label sequences(TimeAlignedLabelSeq) for a set of sequences
    """
    def __init__(self, name, nclass, seq2talabseq):
        self.name = name
        self.nclass = nclass
        self.seq2talabseq = seq2talabseq
    
    def __getitem__(self, seq):
        return self.seq2talabseq[seq]

    def __str__(self):
        return "name=%s, nclass=%s, nseqs=%s" % (
                self.name, self.nclass, len(self.seq2talabseq))

    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            labs = np.concatenate(
                    [talabseq.lablist for talabseq in self.seq2talabseq.values()])
            self._lablist = sorted(np.unique(labs))
        return self._lablist

class Labels(object):
    """
    labels(int) for a set of sequences
    """
    def __init__(self, name, nclass, seq2lab):
        self.name = name
        self.nclass = nclass
        self.seq2lab = seq2lab

    def __getitem__(self, seq):
        return self.seq2lab[seq]

    @property
    def lablist(self):
        if not hasattr(self, "_lablist"):
            self._lablist = sorted(np.unique(self.seq2lab.values()))
        return self._lablist

class SequenceDataset(object):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], min_len=1):
        """
        Args:
            feat_scp(str): feature scp path
            len_scp(str): sequence-length scp path
            lab_specs(list): list of label specifications. each is 
                (name, number of classes, scp path)
            talab_specs(list): list of time-aligned label specifications.
                each is (name, number of classes, ali path)
            min_len(int): keep sequence no shorter than min_len
        """
        feats = scp2dict(feat_scp)
        lens = scp2dict(len_scp, int, feats.keys())

        self.seqlist = [k for k in feats.keys() if lens[k] >= min_len]
        self.feats = OrderedDict([(k, feats[k]) for k in self.seqlist])
        self.lens = OrderedDict([(k, lens[k]) for k in self.seqlist])
        print("%s: %s out of %s kept, min_len = %d" % (
            self.__class__.__name__, len(self.feats), len(feats), min_len))
        
        self.labs_d = OrderedDict()
        for lab_spec in lab_specs:
            name, nclass, seq2lab = load_lab(lab_spec, self.seqlist)
            self.labs_d[name] = Labels(name, nclass, seq2lab)
        self.talabseqs_d = OrderedDict()
        for talab_spec in talab_specs:
            name, nclass, seq2talabs = load_talab(talab_spec, self.seqlist)
            self.talabseqs_d[name] = TimeAlignedLabelSeqs(name, nclass, seq2talabs)

    def iterator(self, bs, lab_names=[], talab_names=[], seqs=None, 
            shuffle=False, rem=True, mapper=None):
        """
        Args:
            bs(int): batch size
            lab_names(list): list of names of labels to include
            talab_names(list): list of names of time-aligned labels to include
            seqs(list): list of sequences to iterate. iterate all if seqs is None
            shuffle(bool): shuffle sequence order if true
            rem(bool): yield remained batch if true
            mapper(callable): feat is mapped by mapper if not None
        Return:
            keys(list): list of sequences(str)
            feats(list): list of feats(str/mapper(str))
            lens(list): list of sequence lengths(int)
            labs(list): list of included labels(int list)
            talabs(list): list of included time-aligned labels(talabel list)
        """
        seqs = self.seqlist if seqs is None else seqs
        mapper = (lambda x: x) if mapper is None else mapper
        if shuffle:
            np.random.shuffle(seqs)

        keys, feats, lens, labs, talabs = [], [], [], [], []
        for seq in seqs:
            keys.append(seq)
            feats.append(mapper(self.feats[seq]))
            lens.append(self.lens[seq])
            labs.append([self.labs_d[name][seq] for name in lab_names])
            talabs.append([self.talabseqs_d[name][seq] for name in talab_names])
            if len(keys) == bs:
                yield keys, feats, lens, labs, talabs
                keys, feats, lens, labs, talabs = [], [], [], [], []
        if rem and bool(keys):
            yield keys, feats, lens, labs, talabs

    def seqs_of_lab(self, lab_name, lab):
        return [seq for seq in self.seqlist if self.labs_d[lab_name][seq] == lab]

    def seqs_of_talab(self, talab_name, lab):
        return [seq for seq in self.seqlist \
                if lab in self.talabseqs_d[talab_name][seq].lablist]

    def get_shape(self, mapper=None):
        raise NotImplementedError

class KaldiDataset(SequenceDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[], 
            min_len=1, preload=False, mvn_path=None):
        """
        Args:
            preload(bool): preload all features into memory if true
        """
        super(KaldiDataset, self).__init__(
                feat_scp, len_scp, lab_specs, talab_specs, min_len)
        if preload:
            with SBFMReader("scp:%s" % feat_scp) as f:
                self.feats = OrderedDict([(k, v) for k, v in f])
        else:
            self.feats = RABFMReader("scp:%s" % feat_scp)

        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                self.mvn_params = self.compute_mvn()
                with open(mvn_path, "wb") as f:
                    cPickle.dump(self.mvn_params, f)
            else:
                with open(mvn_path) as f:
                    self.mvn_params = cPickle.load(f)
        else:
            self.mvn_params = None

    def compute_mvn(self):
        n, x, x2 = 0., 0., 0.
        for seq in self.seqlist:
            feat = self.feats[seq]
            x += np.sum(feat, axis=0, keepdims=True)
            x2 += np.sum(feat ** 2, axis=0, keepdims=True)
            n += feat.shape[0]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean, "std": std}
    
    def apply_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params["mean"]) / self.mvn_params["std"]

    def undo_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params["std"] + self.mvn_params["mean"]

    def iterator(self, bs, lab_names=[], talab_names=[], seqs=None,
            shuffle=False, rem=True, mapper=None):
        if mapper is None:
            new_mapper = self.apply_mvn
        else:
            new_mapper = lambda x: mapper(self.apply_mvn(x))
        return super(KaldiDataset, self).iterator(
                bs, lab_names, talab_names, seqs, shuffle, rem, new_mapper)

    def get_shape(self):
        seq_shape = self.feats[self.seqlist[0]].shape
        return (None,) + tuple(seq_shape[1:])

class NumpyDataset(SequenceDataset):
    def __init__(self, feat_scp, len_scp, lab_specs=[], talab_specs=[],
            min_len=1, preload=False, mvn_path=None):
        super(NumpyDataset, self).__init__(
                feat_scp, len_scp, lab_specs, talab_specs, min_len)
        if preload:
            feats = OrderedDict()
            for seq in self.seqlist:
                with open(self.feats[seq]) as f:
                    feats[seq] = np.load(f)
            self.feats = feats
            print "preloaded features"
        else:
            self.feats = self.feat_getter(self.feats)

        if mvn_path is not None:
            if not os.path.exists(mvn_path):
                self.mvn_params = self.compute_mvn()
                with open(mvn_path, "wb") as f:
                    cPickle.dump(self.mvn_params, f)
            else:
                with open(mvn_path) as f:
                    self.mvn_params = cPickle.load(f)
        else:
            self.mvn_params = None

    class feat_getter:
        def __init__(self, feats):
            self.feats = dict(feats)
            
        def __getitem__(self, seq):
            with open(self.feats[seq]) as f:
                feat = np.load(f)
            return feat

    def compute_mvn(self):
        n, x, x2 = 0., 0., 0.
        for seq in self.seqlist:
            feat = self.feats[seq]
            x += np.sum(feat, axis=0, keepdims=True)
            x2 += np.sum(feat ** 2, axis=0, keepdims=True)
            n += feat.shape[0]
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        return {"mean": mean, "std": std}
    
    def apply_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return (feats - self.mvn_params["mean"]) / self.mvn_params["std"]

    def undo_mvn(self, feats):
        if self.mvn_params is None:
            return feats
        else:
            return feats * self.mvn_params["std"] + self.mvn_params["mean"]

    def iterator(self, bs, lab_names=[], talab_names=[], seqs=None,
            shuffle=False, rem=True, mapper=None):
        if mapper is None:
            new_mapper = self.apply_mvn
        else:
            new_mapper = lambda x: mapper(self.apply_mvn(x))
        return super(NumpyDataset, self).iterator(
                bs, lab_names, talab_names, seqs, shuffle, rem, new_mapper)

    def get_shape(self):
        seq_shape = self.feats[self.seqlist[0]].shape
        return (None,) + tuple(seq_shape[1:])
