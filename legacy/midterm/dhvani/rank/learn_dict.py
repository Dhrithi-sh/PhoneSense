"""Learn English to Hindi translations from the corpus itself.

Jagran's headlines end with an English version of the same headline, e.g.

    भूकंप के झटकों से कांपा उत्तर भारत ... - earthquake tremors felt in delhincr ...

so the frozen corpus contains about 970 Hindi/English headline pairs. They
aren't word-for-word translations, but a Hindi word and its English meaning
keep turning up in the same pairs. We measure that with the Dice coefficient

    dice(e, h) = 2 x pairs with both / (pairs with e + pairs with h)

and keep, for each English word, its best Hindi word if the pair is seen at
least MIN_PAIRS times and the Dice score is at least MIN_DICE. Common words on
either side (Hindi function words, English stop words) are left out first,
because they co-occur with everything.

The result is written in the news dictionary's format to
dhvani/rank/data/en_hi_learned.tsv (word pairs only, no article text), and the
cross-lingual layer loads it under the hand-made dictionary: a hand-made entry
always wins, learned entries fill the gaps (mostly names and news words).

    python -m dhvani.rank.learn_dict          # learn from data/news.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dhvani.rank.xling import ENGLISH_STOP_WORDS

LEARNED_DICT = Path(__file__).resolve().parent / "data" / "en_hi_learned.tsv"
CORPUS = Path(__file__).resolve().parents[2] / "data" / "news.jsonl"

MIN_PAIRS = 3
MIN_DICE = 0.5
COMMON_SHARE = 0.05        # a word in more than 5% of pairs is too common to align
EXTRA_ENGLISH_STOP = {"news", "hindi", "latest", "today", "update", "updates", "live", "says", "said",
                      "after", "amid", "new", "big", "know", "how", "why", "s", "not", "do", "no", "his", "her", "their"}

_TAIL = re.compile(r"\s-\s([a-z0-9][a-z0-9 ,.'&:/-]*)$")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]+")
_ENGLISH = re.compile(r"[a-z]+")


def split_headline(headline):
    """(hindi part, english part) for a headline with an English version appended, else None."""
    m = _TAIL.search(headline)
    if not m:
        return None
    return headline[:m.start()], m.group(1)


def headline_pairs(articles):
    """[(set of Hindi words, set of English words)] from articles' headlines."""
    pairs = []
    for article in articles:
        split = split_headline(article.get("headline", ""))
        if not split:
            continue
        hindi = {w for w in _DEVANAGARI.findall(split[0]) if len(w) > 1}
        english = {w for w in _ENGLISH.findall(split[1].lower())
                   if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in EXTRA_ENGLISH_STOP}
        if hindi and english:
            pairs.append((hindi, english))
    return pairs


def learn(pairs, min_pairs=MIN_PAIRS, min_dice=MIN_DICE, common_share=COMMON_SHARE):
    """{english: (hindi, dice, pairs_with_both)}: the best Hindi word for each English word."""
    n = len(pairs)
    ce, ch, ceh = Counter(), Counter(), Counter()
    for hindi, english in pairs:
        ce.update(english)
        ch.update(hindi)
        for e in english:
            for h in hindi:
                ceh[(e, h)] += 1
    too_common = common_share * n
    best = {}
    for (e, h), both in ceh.items():
        if both < min_pairs or ce[e] > too_common or ch[h] > too_common:
            continue
        dice = 2 * both / (ce[e] + ch[h])
        if dice >= min_dice and (e not in best or dice > best[e][1]):
            best[e] = (h, dice, both)
    return best


def write_dict(learned, path=LEARNED_DICT):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# English to Hindi pairs learned from Jagran's bilingual headlines (dhvani/rank/learn_dict.py).\n")
        f.write("# Same format as en_hi_news.tsv. The comment above each entry gives its Dice score\n")
        f.write("# and how many headline pairs had both words.\n")
        for e in sorted(learned):
            h, dice, both = learned[e]
            f.write(f"# dice {dice:.2f}, {both} pairs\n{e}\t{h}\n")
    return path


def read_articles(path=CORPUS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Learn English to Hindi pairs from bilingual headlines.")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--out", default=LEARNED_DICT)
    parser.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    parser.add_argument("--min-dice", type=float, default=MIN_DICE)
    args = parser.parse_args(argv)
    pairs = headline_pairs(read_articles(args.corpus))
    learned = learn(pairs, args.min_pairs, args.min_dice)
    path = write_dict(learned, args.out)
    print(f"{len(pairs)} headline pairs, {len(learned)} English words learned -> {path}")


if __name__ == "__main__":
    main()
