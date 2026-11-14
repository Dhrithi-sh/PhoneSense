from dhvani.eval.ltr import FEATURES, Logistic, leave_one_need_out


def _row(**kw):
    return {f: kw.get(f, 0.0) for f in FEATURES}


def test_logistic_learns_the_feature_that_separates_relevance():
    rows = [_row(bm25=x, cosine=0.5) for x in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9)]
    model = Logistic(steps=500).fit(rows, [0, 0, 0, 1, 1, 1])
    assert model.weights()["bm25"] > 0
    assert model.score(_row(bm25=0.9, cosine=0.5)) > model.score(_row(bm25=0.1, cosine=0.5))


def _need(name, good):
    # the relevant article always has a high zone value, and the net order puts it last
    rows = [(f"{name}_d{i}", _row(zone=0.1, cosine=0.9 - i * 0.1), 0) for i in range(3)]
    rows.append((f"{name}_good", _row(zone=good, cosine=0.1), 2))
    return name, rows


def test_leave_one_need_out_beats_a_bad_base_order():
    table = {f"{n}_hi": _need(n, 0.9) for n in ("A", "B", "C", "D")}
    net, learned, weights = leave_one_need_out(table)
    assert set(net) == set(learned) == set(table)
    assert all(learned[q][0] > net[q][0] for q in table)
    assert weights["zone"] > 0
