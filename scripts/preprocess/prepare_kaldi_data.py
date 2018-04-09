"""
extracting features from wav.scp and save in ark
"""
import os
import sys
import argparse
import subprocess

def _get_kaldi_root():
    KALDI_ROOT = os.environ.get("KALDI_ROOT")
    if KALDI_ROOT is None:
        raise ValueError("KALDI_ROOT not found in environment variables")
    print "KALDI_ROOT=%s" % KALDI_ROOT
    return KALDI_ROOT
    
def _maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

def _run_proc(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))

def main(wav_scp, feat_ark, feat_scp, len_scp, fbank_conf):
    KALDI_ROOT = _get_kaldi_root()

    _maybe_makedir(os.path.dirname(feat_ark))
    _maybe_makedir(os.path.dirname(feat_scp))
    _maybe_makedir(os.path.dirname(len_scp))

    cmd = [os.path.join(KALDI_ROOT, "src/featbin/compute-fbank-feats")]
    cmd += ["--config=%s" % fbank_conf]
    cmd += ["scp,p:%s" % wav_scp, "ark,scp:%s,%s" % (feat_ark, feat_scp)]
    _run_proc(cmd)
    
    cmd = [os.path.join(KALDI_ROOT, "src/featbin/feat-to-len")]
    cmd += ["scp:%s" % feat_scp, "ark,t:%s" % len_scp]
    _run_proc(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="input wav scp file")
    parser.add_argument("feat_ark", type=str, help="output feats.ark file")
    parser.add_argument("feat_scp", type=str, help="output feats.scp file")
    parser.add_argument("len_scp", type=str, help="output len.scp file")
    parser.add_argument("--fbank_conf", type=str, default="./misc/fbank.conf",
            help="kaldi fbank configuration")
    args = parser.parse_args()
    print args

    main(args.wav_scp, args.feat_ark, args.feat_scp, args.len_scp, args.fbank_conf)
