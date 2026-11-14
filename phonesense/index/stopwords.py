"""Built-in English stopword list (kept in the repo, so no corpus download is needed).

Adapted from our midterm project Dhvani-lingual (`text/stopwords.py`): the Hindi list was
replaced by an English one plus a protected set.

Based on the common SMART / NLTK English lists, trimmed to function words. Words that carry
meaning in phone search (``5g``, ``4g``, ``pro``, ``max``, ``plus``, ``mini``, ``ultra``,
``lite``) are listed in :data:`PROTECTED` and are never removed, even if a future edit adds
them to :data:`STOPWORDS`.
"""

from __future__ import annotations

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how how's
    i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves out over own same
    shan't she she'd she'll she's should shouldn't so some such than that that's the their
    theirs them themselves then there there's these they they'd they'll they're they've this
    those through to too under until up very was wasn't we we'd we'll we're we've were
    weren't what what's when when's where where's which while who who's whom why why's will
    with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
    also just s t d ll m o re ve y
    """.split()
)

# Never removed as stopwords and never stemmed (they are part of model names).
PROTECTED: frozenset[str] = frozenset({"5g", "4g", "pro", "max", "plus", "mini", "ultra", "lite"})
