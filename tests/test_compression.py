import random

import pytest

from phonesense.index import Analyzer, InvertedIndex, Posting
from phonesense.index.postings import (
    decode_postings,
    encode_postings,
    gaps,
    ungaps,
    vb_decode,
    vb_encode,
    vb_encode_number,
)


def test_vb_known_values():
    # Manning et al., IIR table 5.4 (stop bit on the last byte)
    assert vb_encode_number(5) == bytes([0b10000101])
    assert vb_encode_number(824) == bytes([0b00000110, 0b10111000])
    assert vb_encode_number(214577) == bytes([0b00001101, 0b00001100, 0b10110001])
    assert vb_encode([0]) == bytes([0x80])
    assert vb_encode([127]) == bytes([0xFF])
    assert vb_encode([128]) == bytes([0x01, 0x80])


EDGE = [0, 1, 127, 128, 129, 16383, 16384, 2**21 - 1, 2**21, 2**31 - 1, 2**32, 2**40, 2**62]


@pytest.mark.parametrize("seed", range(5))
def test_vb_round_trip_random(seed):
    rng = random.Random(seed)
    nums = EDGE + [
        rng.choice([rng.randrange(128), rng.randrange(2**14), rng.randrange(2**35)])
        for _ in range(2000)
    ]
    rng.shuffle(nums)
    assert vb_decode(vb_encode(nums)) == nums


def test_vb_huge_and_empty():
    assert vb_decode(vb_encode([2**70, 3])) == [2**70, 3]
    assert vb_decode(b"") == []
    with pytest.raises(ValueError):
        vb_encode([-1])
    with pytest.raises(ValueError):
        vb_decode(bytes([0x01]))  # no stop bit


def test_gaps():
    assert gaps([3, 7, 8, 200]) == [3, 4, 1, 192]
    assert ungaps(gaps([0, 5, 6])) == [0, 5, 6]


@pytest.mark.parametrize("compression", ["vbyte", "none"])
def test_postings_round_trip(compression):
    rng = random.Random(1)
    docs = sorted(rng.sample(range(100_000), 300))
    plist = []
    for d in docs:
        pos = sorted(rng.sample(range(5000), rng.randint(1, 20)))
        plist.append(Posting(d, len(pos), pos))
    data = encode_postings(plist, compression)
    assert decode_postings(data, len(plist), compression) == plist


def test_vbyte_smaller_than_plain():
    plist = [Posting(d, 2, [d % 50, d % 50 + 3]) for d in range(0, 3000, 3)]
    assert len(encode_postings(plist, "vbyte")) < len(encode_postings(plist, "none")) / 3


def test_decoded_index_equals_uncompressed(make_phone, tmp_path):
    rng = random.Random(7)
    words = [f"w{i}" for i in range(40)] + ["camera", "battery", "128gb", "5g"]
    phones = [
        make_phone(
            f"P{i:03d}",
            " ".join(rng.choices(words, k=6)),
            [" ".join(rng.choices(words, k=rng.randint(0, 30)))],
            " ".join(rng.choices(words, k=rng.randint(0, 200))),
        )
        for i in range(60)
    ]
    an = Analyzer()
    idx = InvertedIndex.build(phones, an)
    sizes_vb = idx.save(tmp_path / "vb", compression="vbyte")
    sizes_raw = idx.save(tmp_path / "raw", compression="none")
    vb, raw = InvertedIndex.load(tmp_path / "vb", an), InvertedIndex.load(tmp_path / "raw", an)
    assert vb == raw == idx
    assert sizes_vb["text.post"] < sizes_raw["text.post"]
