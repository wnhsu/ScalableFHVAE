"""
prepare LibriSpeech data for FHVAE
"""
import os
import wave
import argparse
import subprocess

def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("librispeech_dir", type=str, help="LibriSpeech raw data directory")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/librispeech_np_fbank",
        help="output data directory")
parser.add_argument("--train_list", type=str, nargs="*", default=["train-clean-100",],
        help="train sets to include {train-clean-100, train-clean-360, train-other-500}")
parser.add_argument("--dev_list", type=str, nargs="*", default=["dev-clean", "dev-other"],
        help="dev sets to include {dev-clean, dev-other}")
parser.add_argument("--test_list", type=str, nargs="*", default=["test-clean", "dev-other"],
        help="test sets to include {test-clean, test-other}")
args = parser.parse_args()
print args

# dump wav scp
def find_audios(d):
    uid_path = []
    for root, _, fnames in sorted(os.walk(d)):
        for fname in fnames:
            if fname.endswith(".flac") or fname.endswith(".FLAC"):
                uid_path.append((os.path.splitext(fname)[0], "%s/%s" % (root, fname)))
    return sorted(uid_path, key=lambda x: x[0])

def write_scp(out_path, set_list):
    maybe_makedir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        uid_path = []
        for s in set_list:
            uid_path += find_audios("%s/%s" % (args.librispeech_dir, s))
        for uid, path in uid_path:
            f.write("%s %s\n" % (uid, path))

write_scp("%s/train/wav.scp" % args.out_dir, args.train_list)
write_scp("%s/dev/wav.scp" % args.out_dir, args.dev_list)
write_scp("%s/test/wav.scp" % args.out_dir, args.test_list)

print "generated wav scp"

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
