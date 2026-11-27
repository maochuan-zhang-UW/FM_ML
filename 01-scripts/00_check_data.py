"""Step 0: check that the input data are in place and unchanged.

Every input of the pipeline and figures is listed in data_manifest.csv
(path relative to the repository root, size, SHA-256, what uses it). Download
the data archive (see README, Data availability), unpack it in the
repository root so that 02-data/raw/, 02-data/archive/ and 07-files/ exist,
then run

  python 01-scripts/00_check_data.py           # sizes and checksums
  python 01-scripts/00_check_data.py --quick   # sizes only

The results archive (step 5-8 outputs) is listed in results_manifest.csv:

  python 01-scripts/00_check_data.py --manifest results_manifest.csv
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="check sizes only")
    ap.add_argument("--manifest", default="data_manifest.csv", help="manifest, relative to the repository root")
    a = ap.parse_args()
    bad = 0
    rows = list(csv.DictReader(open(ROOT / a.manifest)))
    for r in rows:
        p = ROOT / r["path"]
        if not p.exists():
            msg = "MISSING"
        elif p.stat().st_size != int(r["bytes"]):
            msg = "WRONG SIZE"
        elif not a.quick and sha256(p) != r["sha256"]:
            msg = "WRONG CHECKSUM"
        else:
            continue
        bad += 1
        print(f"{msg:15s} {r['path']}  ({r['used_for']})")
    print(f"{len(rows) - bad} of {len(rows)} files OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
