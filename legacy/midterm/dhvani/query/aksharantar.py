"""Load the Aksharantar Hindi dataset and export the pairs this layer needs.

Aksharantar (`ai4bharat/Aksharantar`) ships one zip per language at the repo
root; Hindi is ``hin.zip`` (~33 MB), containing ``hin_train.json``,
``hin_test.json`` and ``hin_valid.json`` as JSON-lines. Each row has a
Devanagari ``native word`` and its human romanisation ``english word``, e.g.::

    {"native word": "मौसम", "english word": "mausam", "source": "...", ...}

From it we build two files under ``data/aksharantar/`` (git-ignored):

* **train_pairs.tsv** — ``(canonical, human)`` where ``canonical =
  romanize(native word)`` and ``human`` is the dataset's romanisation. This
  teaches the learned edit distance which Roman spelling changes are common.
* **test_pairs.tsv** — ``(native word, human romanisation)`` for the word-level
  matching evaluation.

The old ``load_dataset("ai4bharat/Aksharantar", "hin")`` config no longer
exists, so we pull the zip directly with ``huggingface_hub`` and read it.

Run once::

    python -m dhvani.query.aksharantar --max-train 50000 --max-test 5000
"""

import argparse
import json
import os
import zipfile

from dhvani.query.roman import romanize

OUT_DIR = os.path.join("data", "aksharantar")
REPO_ID = "ai4bharat/Aksharantar"
ZIP_NAME = "hin.zip"


def _zip_path():
    from huggingface_hub import hf_hub_download  # lazy import; only needed here

    return hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset")


def read_split(split, limit=None):
    """Yield ``(native_word, human_romanisation)`` from ``hin_<split>.json``.

    Reads straight out of the zip so the 195 MB train file never hits disk.
    ``split`` is one of ``"train"``, ``"test"``, ``"valid"``.
    """
    rows = []
    with zipfile.ZipFile(_zip_path()) as z:
        with z.open(f"hin_{split}.json") as fh:
            for raw in fh:
                row = json.loads(raw)
                native = (row.get("native word") or "").strip()
                roman = (row.get("english word") or "").strip().lower()
                if native and roman:
                    rows.append((native, roman))
                if limit and len(rows) >= limit:
                    break
    return rows


def _write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for a, b in rows:
            fh.write(f"{a}\t{b}\n")


def load_pairs(path):
    """Read an ``a<TAB>b`` TSV written by this module; returns ``[(a, b), ...]``."""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export Aksharantar Hindi pairs.")
    ap.add_argument("--max-train", type=int, default=50000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    train = read_split("train", args.max_train)
    train_pairs = [(romanize(native), roman) for native, roman in train]
    train_path = os.path.join(args.out_dir, "train_pairs.tsv")
    _write_tsv(train_path, train_pairs)
    print(f"wrote {len(train_pairs)} training pairs -> {train_path}")

    test = read_split("test", args.max_test)
    test_path = os.path.join(args.out_dir, "test_pairs.tsv")
    _write_tsv(test_path, test)  # (native word, human romanisation)
    print(f"wrote {len(test)} test pairs -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
