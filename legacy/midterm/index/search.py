from math import isqrt

from index.positional import Index


def build_skip_pointers(postings):
    """
    Build skip pointers for a postings list.

    A postings list contains tuples of:

        (doc_id, tf, positions)

    Skip pointers are represented as:

        {current_index: target_index}

    The skip distance is approximately sqrt(n).
    """

    n = len(postings)

    if n < 2:
        return {}

    step = max(1, isqrt(n))

    skips = {}

    for index in range(0, n - step, step):
        skips[index] = index + step

    return skips


def intersect_postings(left, right):
    """
    Intersect two sorted postings lists using skip pointers.

    Returns the document IDs present in both lists.
    """

    if not left or not right:
        return []

    left_skips = build_skip_pointers(left)
    right_skips = build_skip_pointers(right)

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        left_doc = left[i][0]
        right_doc = right[j][0]

        if left_doc == right_doc:
            result.append(left_doc)
            i += 1
            j += 1

        elif left_doc < right_doc:
            skip_to = left_skips.get(i)

            if (
                skip_to is not None
                and skip_to < len(left)
                and left[skip_to][0] <= right_doc
            ):
                i = skip_to
            else:
                i += 1

        else:
            skip_to = right_skips.get(j)

            if (
                skip_to is not None
                and skip_to < len(right)
                and right[skip_to][0] <= left_doc
            ):
                j = skip_to
            else:
                j += 1

    return result


def and_search(
    idx: Index,
    terms: list[str],
    zone: str = "body",
) -> list[str]:
    """
    Return documents containing all supplied terms.

    The smallest postings list is processed first.
    Skip pointers are used while intersecting postings lists.
    """

    if not terms:
        return []

    posting_lists = []

    for term in terms:
        postings = idx.postings_for(term, zone)

        if not postings:
            return []

        posting_lists.append(postings)

    # Process the smallest postings list first.
    posting_lists.sort(key=len)

    result = posting_lists[0]

    for postings in posting_lists[1:]:
        matching_doc_ids = intersect_postings(
            result,
            postings,
        )

        if not matching_doc_ids:
            return []

        # Convert matching document IDs back into postings.
        result = [
            posting
            for posting in result
            if posting[0] in matching_doc_ids
        ]

    return [
        posting[0]
        for posting in result
    ]


def phrase_search(
    idx: Index,
    terms: list[str],
    zone: str = "body",
) -> list[str]:
    """
    Return documents where all terms occur consecutively
    in the supplied order.
    """

    if not terms:
        return []

    if len(terms) == 1:
        return sorted(
            doc_id
            for doc_id, _, _ in idx.postings_for(
                terms[0],
                zone,
            )
        )

    postings_by_term = []

    for term in terms:
        postings = idx.postings_for(
            term,
            zone,
        )

        if not postings:
            return []

        postings_by_term.append(
            {
                doc_id: positions
                for doc_id, _, positions in postings
            }
        )

    candidate_docs = set(
        postings_by_term[0]
    )

    for postings in postings_by_term[1:]:
        candidate_docs &= set(postings)

    matches = []

    for doc_id in candidate_docs:
        first_positions = postings_by_term[0][doc_id]

        for start_position in first_positions:
            matches_phrase = True

            for offset, postings in enumerate(
                postings_by_term[1:],
                start=1,
            ):
                if (
                    start_position + offset
                    not in postings[doc_id]
                ):
                    matches_phrase = False
                    break

            if matches_phrase:
                matches.append(doc_id)
                break

    return sorted(matches)


def proximity_search(
    idx: Index,
    terms: list[str],
    distance: int,
    zone: str = "body",
) -> list[str]:
    """
    Return documents where all supplied terms occur within
    the specified positional distance.

    The order of the terms does not matter.

    Example:

        proximity_search(
            idx,
            ["दिल्ली", "बारिश"],
            3,
        )

    matches documents where occurrences of "दिल्ली" and
    "बारिश" are at most 3 positions apart.

    Args:
        idx:
            Positional index.

        terms:
            Terms that must occur near each other.

        distance:
            Maximum allowed positional distance.

        zone:
            Index zone, either "headline" or "body".
    """

    if not terms:
        return []

    if distance < 0:
        raise ValueError(
            "distance must be non-negative"
        )

    # Retrieve positional information for every term.
    postings_by_term = []

    for term in terms:
        postings = idx.postings_for(
            term,
            zone,
        )

        if not postings:
            return []

        postings_by_term.append(
            {
                doc_id: positions
                for doc_id, _, positions in postings
            }
        )

    # Only documents containing every term can match.
    candidate_docs = set(
        postings_by_term[0]
    )

    for postings in postings_by_term[1:]:
        candidate_docs &= set(postings)

    matches = []

    for doc_id in candidate_docs:
        all_positions = []

        for postings in postings_by_term:
            all_positions.append(
                postings[doc_id]
            )

        # Check whether there is one occurrence of each
        # term inside the requested distance window.
        #
        # For each possible anchor position, calculate
        # the closest occurrence of every other term.
        for anchor in all_positions[0]:

            if all(
                any(
                    abs(
                        anchor - position
                    ) <= distance
                    for position in positions
                )
                for positions in all_positions[1:]
            ):
                matches.append(doc_id)
                break

    return sorted(matches)