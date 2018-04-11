"""
prepare TIMIT data for FHVAE
"""
import os
import wave
import argparse
import subprocess
from sphfile import SPHFile

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

# sphfile write_wav is not compatible with python2
def write_wav(fname, sph):
    fh = wave.open(fname,'wb')
    params = (
        sph.format['channel_count'], 
        sph.format['sample_n_bytes'], 
        sph.format['sample_rate'],
        0, 
        'NONE', 'NONE'
    )
    fh.setparams(params)
    data = sph.content
    fh.writeframes( data.tostring() )
    fh.close()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("timit_dir", type=str, help="TIMIT raw data directory")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/timit_np_fbank",
        help="output data directory")
parser.add_argument("--dev_spk", type=str, default="./misc/timit_dev_spk.list",
        help="path to list of dev set speakers")
parser.add_argument("--test_spk", type=str, default="./misc/timit_test_spk.list",
        help="path to list of test set speakers")
args = parser.parse_args()
print args

# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip().lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip().lower() for line in f]

# convert sph to wav and dump scp
wav_dir = os.path.abspath("%s/wav" % args.out_dir)
tr_scp = "%s/train/wav.scp" % args.out_dir
dt_scp = "%s/dev/wav.scp" % args.out_dir
tt_scp = "%s/test/wav.scp" % args.out_dir

maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

paths = []
for root, _, fnames in sorted(os.walk(args.timit_dir)):
    spk = root.split("/")[-1].lower()
    if spk in dt_spks:
        f = dt_f
    elif spk in tt_spks:
        f = tt_f
    else:
        f = tr_f

    for fname in fnames:
        if fname.endswith(".wav") or fname.endswith(".WAV"):
            sph_path = "%s/%s" % (root, fname)
            path = "%s/%s_%s" % (wav_dir, spk, fname)
            uttid = "%s_%s" % (spk, os.path.splitext(fname)[0])
            f.write("%s %s\n" % (uttid, path))
            sph = SPHFile(sph_path)
            write_wav(path, sph)
            
tr_f.close()
dt_f.close()
tt_f.close()

print "converted to wav and dumped scp files"

# compute feature
feat_dir = os.path.abspath("%s/%s" % (args.out_dir, args.ftype))
maybe_makedir(feat_dir)

def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=%s" % args.ftype]
    cmd += ["%s/%s/wav.scp" % (args.out_dir, name), feat_dir]
    cmd += ["%s/%s/feats.scp" % (args.out_dir, name)]
    cmd += ["%s/%s/len.scp" % (args.out_dir, name)]
    
    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))

for name in ["train", "dev", "test"]:
    compute_feature(name)

print "computed feature"
