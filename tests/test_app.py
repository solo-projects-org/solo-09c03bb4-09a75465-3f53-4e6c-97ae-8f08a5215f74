"""Unit and HTTP-level tests for the seal census review app."""

from __future__ import annotations

import itertools
import os
import random
import tempfile

import pytest

os.environ.setdefault(
    "SEAL_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="seal-test-"), "seals.db"),
)

from app.model import (  # noqa: E402
    Dataset,
    InputError,
    Verdicts,
    apply_verdict_changes,
    chapman,
    maximum_bipartite_matching,
    parse_ids_text,
    parse_pairs_text,
    recommend,
)


# ------------------------------------------------------------------ helpers

def brute_force_m_range(ds: Dataset) -> tuple[int, int]:
    """Enumerate every subset of candidate pairs and keep legal matchings."""
    forced = ds.verdicts.same
    banned = ds.verdicts.diff
    open_pairs = [
        c.key for c in ds.candidates if c.key not in forced and c.key not in banned
    ]
    counts = []
    for r in range(len(open_pairs) + 1):
        for combo in itertools.combinations(open_pairs, r):
            chosen = list(forced) + list(combo)
            left = [a for a, _ in chosen]
            right = [b for _, b in chosen]
            if len(set(left)) == len(left) and len(set(right)) == len(right):
                counts.append(len(chosen))
    return min(counts), max(counts)


def make(n1, n2, pairs, verdicts=None):
    return Dataset.from_raw(
        [f"L{i}" for i in range(n1)],
        [f"R{i}" for i in range(n2)],
        [(f"L{a}", f"R{b}") for a, b in pairs],
        verdicts,
    )


def test_chapman_monotone():
    assert chapman(10, 10, 0) == 120.0
    assert chapman(10, 10, 2) == pytest.approx(121 / 3 - 1)
    assert chapman(1, 1, 1) == 1.0


def test_estimate_matches_brute_force_random_graphs():
    rng = random.Random(20261122)
    for trial in range(40):
        n1, n2 = rng.randint(2, 5), rng.randint(2, 5)
        all_pairs = [(a, b) for a in range(n1) for b in range(n2)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[: rng.randint(1, len(all_pairs) // 2 + 1)]
        ds = make(n1, n2, pairs)
        est = ds.estimate()
        lo, hi = brute_force_m_range(ds)
        assert (est.m_min, est.m_max) == (lo, hi), (trial, pairs, est)
        assert est.n_low <= est.n_high + 1e-9
        # every legal matching's estimate lies inside the reported interval
        for m in range(lo, hi + 1):
            value = chapman(n1, n2, m)
            assert est.n_low - 1e-9 <= value <= est.n_high + 1e-9


def test_estimate_with_verdicts_matches_brute_force():
    rng = random.Random(7)
    for _ in range(30):
        n1, n2 = rng.randint(3, 5), rng.randint(3, 5)
        all_pairs = [(a, b) for a in range(n1) for b in range(n2)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[: rng.randint(3, 9)]
        # pick a legal forced matching: disjoint pairs
        rng.shuffle(pairs)
        forced, used_a, used_b = set(), set(), set()
        for a, b in pairs:
            if a in used_a or b in used_b:
                continue
            if rng.random() < 0.5:
                forced.add((f"L{a}", f"R{b}"))
                used_a.add(a)
                used_b.add(b)
        remaining = [p for p in pairs if (f"L{p[0]}", f"R{p[1]}") not in forced]
        diff = {
            (f"L{a}", f"R{b}") for a, b in remaining if rng.random() < 0.4
        }
        ds = make(n1, n2, pairs, Verdicts(forced, diff))
        est = ds.estimate()
        lo, hi = brute_force_m_range(ds)
        assert len(forced) <= lo
        assert (est.m_min, est.m_max) == (lo, hi)


def test_candidate_contention_caps_matching():
    # Two round-1 IDs both only see the same round-2 ID: only one can match.
    ds = make(2, 1, [(0, 0), (1, 0)])
    est = ds.estimate()
    assert (est.m_min, est.m_max) == (0, 1)


def test_contention_chain_limits_maximum_matching():
    # L0 can take R0/R1, L1 can take R0 only: maximum is 2.  Forcing L0-R0
    # blocks L1-R0, so that trial freezes m at 1; forcing L0-R1 still leaves
    # L1-R0 free, so m can reach 2.
    ds = make(2, 2, [(0, 0), (0, 1), (1, 0)])
    assert (ds.estimate().m_min, ds.estimate().m_max) == (0, 2)
    trials = {t.candidate.key: t for t in recommend(ds).trials}
    trial = trials[("L0", "R0")]
    assert (trial.same.m_min, trial.same.m_max) == (1, 1)
    assert (trial.diff.m_min, trial.diff.m_max) == (0, 2)
    trial_alt = trials[("L0", "R1")]
    assert (trial_alt.same.m_min, trial_alt.same.m_max) == (1, 2)
    trial_leaf = trials[("L1", "R0")]
    assert (trial_leaf.same.m_min, trial_leaf.same.m_max) == (1, 2)
    # L0-R0 is the only pair that freezes m at 1, so its "same" trial has zero
    # width, while the other two keep an interval.  The decision width is the
    # wider of the two trials, which makes L0-R1 and L1-R0 the best pair.
    assert trial.same_width == 0.0
    assert trial_alt.same_width > 0.0
    assert recommend(ds).best_width == trial_alt.effective_width
    assert len(recommend(ds).best) == 2


def test_opencandidates_blocked_after_same_verdict():
    ds = make(2, 2, [(0, 0), (0, 1), (1, 1)], Verdicts({("L1", "R1")}, set()))
    assert {c.key for c in ds.matchable_open_candidates()} == {("L0", "R0")}
    assert {c.key for c in ds.blocked_open_candidates()} == {("L0", "R1")}


def test_invalid_id_characters_rejected():
    with pytest.raises(InputError, match="不支持的字符"):
        Dataset.from_raw(["照片一"], ["B1"], [("照片一", "B1")])


def test_same_verdict_blocks_its_ids_from_other_pairs():
    ds = make(2, 2, [(0, 0), (0, 1), (1, 1)], Verdicts({("L0", "R1")}, set()))
    est = ds.estimate()
    # L0-R1 forced; L0 cannot also take R0, so free edge L1-R1 is blocked
    # by R1 and L0-R0 is blocked by L0; m is fixed at 1.
    assert (est.m_min, est.m_max) == (1, 1)


def test_diff_verdict_removes_edge():
    ds = make(2, 2, [(0, 0), (1, 1)], Verdicts(set(), {("L0", "R0")}))
    est = ds.estimate()
    assert (est.m_min, est.m_max) == (0, 1)


def test_pair_trials_are_consistent_with_definitions():
    ds = make(3, 3, [(0, 0), (1, 1), (2, 2)])
    rec = recommend(ds)
    assert rec.considered == 3
    by_key = {(t.candidate.round1_id, t.candidate.round2_id): t for t in rec.trials}
    t = by_key[("L0", "R0")]
    # same trial: forced pair plus the other two still free -> m 1..3
    assert (t.same.m_min, t.same.m_max) == (1, 3)
    # diff trial: the other two remain -> m 0..2
    assert (t.diff.m_min, t.diff.m_max) == (0, 2)
    # every still-matchable pair has equal widths on the complete graph
    assert rec.best and len(rec.best) == 3


def test_recommendation_prefers_narrowing_pair():
    # A sparse leaf pair whose resolution collapses the interval the most.
    ds = make(3, 3, [(0, 0), (1, 0), (2, 1)])
    rec = recommend(ds)
    widths = {t.candidate.key: t.effective_width for t in rec.trials}
    best = min(widths, key=widths.get)
    assert rec.best[0].candidate.key == best
    assert all(rec.best[0].effective_width <= w + 1e-9 for w in widths.values())


def test_recommendation_excludes_blocked_open_pairs():
    ds = make(2, 2, [(0, 0), (0, 1), (1, 0)], Verdicts({("L0", "R1")}, set()))
    rec = recommend(ds)
    keys = {t.candidate.key for t in rec.trials}
    assert ("L0", "R0") not in keys and ("L0", "R1") not in keys
    assert keys == {("L1", "R0")}


def test_verdict_conflict_on_id_reuse():
    ds = make(2, 2, [(0, 0), (0, 1)])
    merged, msg = apply_verdict_changes(
        ds, [("L0", "R0", "same"), ("L0", "R1", "same")]
    )
    assert merged is None
    assert "冲突" in msg or "最多" in msg
    assert ds.verdicts.same == set()


def test_verdict_conflict_on_relabel():
    ds = make(1, 1, [(0, 0)], Verdicts(set(), {("L0", "R0")}))
    merged, msg = apply_verdict_changes(ds, [("L0", "R0", "same")])
    assert merged is None and "两只" in msg


def test_verdict_idempotent_resubmission():
    ds = make(1, 1, [(0, 0)], Verdicts({("L0", "R0")}, set()))
    merged, msg = apply_verdict_changes(ds, [("L0", "R0", "same")])
    assert merged is not None and merged.same == {("L0", "R0")}


def test_verdict_unknown_pair_rejected():
    ds = make(1, 1, [(0, 0)])
    merged, msg = apply_verdict_changes(ds, [("L9", "R0", "same")])
    assert merged is None and "不在候选清单" in msg


def test_verdict_batch_atomic_on_conflict():
    ds = make(2, 2, [(0, 0), (0, 1)])
    merged, msg = apply_verdict_changes(
        ds, [("L0", "R1", "diff"), ("L0", "R0", "same"), ("L0", "R1", "same")]
    )
    assert merged is None
    assert ds.verdicts.diff == set() and ds.verdicts.same == set()


def test_input_validation_limits_and_duplicates():
    with pytest.raises(InputError, match="重复"):
        Dataset.from_raw(["A", "A"], ["B"], [("A", "B")])
    with pytest.raises(InputError, match="上限"):
        Dataset.from_raw([f"A{i}" for i in range(81)], ["B"], [("A0", "B")])
    with pytest.raises(InputError, match="不在第一轮"):
        Dataset.from_raw(["A"], ["B"], [("X", "B")])
    with pytest.raises(InputError, match="重复出现"):
        Dataset.from_raw(["A"], ["B"], [("A", "B"), ("A", "B")])
    many = [(f"A{i}", f"B{i}") for i in range(81)]
    with pytest.raises(InputError, match="上限"):
        Dataset.from_raw(
            [f"A{i}" for i in range(81)], [f"B{i}" for i in range(81)], many
        )
    # Candidate cap: 801 distinct pairs within the per-round ID cap.
    ids = [f"A{i:02d}" for i in range(80)]
    other = [f"B{i:02d}" for i in range(80)]
    too_many = [(a, b) for a in ids[:20] for b in other[:41]][:801]
    with pytest.raises(InputError, match="上限"):
        Dataset.from_raw(ids, other, too_many)


def test_parse_text_formats():
    assert parse_ids_text("A1 A2\nA3\n", "第一轮") == ["A1", "A2", "A3"]
    assert parse_pairs_text("A1,B1\nA2 -> B2\nA3\tB3\n") == [
        ("A1", "B1"),
        ("A2", "B2"),
        ("A3", "B3"),
    ]
    with pytest.raises(InputError, match="两个编号"):
        parse_pairs_text("A1 B1 C1")


def test_matching_respects_skip_parameters():
    edges = [("a", "x"), ("a", "y"), ("b", "x")]
    assert maximum_bipartite_matching(edges) == 2
    from app.model import BipartiteGraph

    g = BipartiteGraph(edges)
    assert g.match_count_without("a", "x") == 0  # only a-x/a-y/b-x exist
    assert g.match_count_without_edge("a", "x") == 2


# --------------------------------------------------------------------- HTTP

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SEAL_DB_PATH", str(tmp_path / "http.db")
    )
    from fastapi.testclient import TestClient

    from app import db
    from app.main import app

    db.init_db()
    with TestClient(app) as c:
        yield c


def test_http_create_review_flow(client):
    r = client.post(
        "/surveys",
        data={
            "name": "测试区",
            "round1_text": "A1\nA2\nA3",
            "round2_text": "B1\nB2",
            "pairs_text": "A1,B1\nA1,B2\nA2,B1",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/surveys/")

    page = client.get(loc)
    assert page.status_code == 200
    assert "推荐" in page.text

    api = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    assert api["estimate"]["n1"] == 3 and api["estimate"]["n2"] == 2
    assert api["estimate"]["m_min"] == 0 and api["estimate"]["m_max"] == 2
    assert api["recommendation"]["considered"] == 3

    # confirm the recommended pair as same
    best = api["recommendation"]["best"][0]
    r2 = client.post(
        f"{loc}/verdicts",
        data={f"verdict__0": "same"},
        follow_redirects=False,
    )
    assert r2.status_code == 303
    assert "error" not in r2.headers["location"]
    after = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    assert after["estimate"]["m_min"] >= 1


def test_http_conflict_keeps_state(client):
    r = client.post(
        "/surveys",
        data={
            "name": "冲突区",
            "round1_text": "A1\nA2",
            "round2_text": "B1\nB2",
            "pairs_text": "A1,B1\nA1,B2",
        },
        follow_redirects=False,
    )
    loc = r.headers["location"]
    ok = client.post(
        f"{loc}/verdicts", data={"verdict__0": "same"}, follow_redirects=False
    )
    assert ok.status_code == 303 and "error" not in ok.headers["location"]
    before = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    bad = client.post(
        f"{loc}/verdicts", data={"verdict__1": "same"}, follow_redirects=False
    )
    assert bad.status_code == 303 and "error=" in bad.headers["location"]
    after = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    assert after == before
    from urllib.parse import unquote

    assert "冲突" in unquote(bad.headers["location"])


def test_http_invalid_upload_shows_error(client):
    r = client.post(
        "/surveys",
        data={
            "name": "坏数据",
            "round1_text": "A1\nA1",
            "round2_text": "B1",
            "pairs_text": "A1,B1",
        },
    )
    assert r.status_code == 400
    assert "重复" in r.text


def test_sample_page_prefills(client):
    r = client.get("/samples/small")
    assert r.status_code == 200
    assert "B01,C" in r.text or "B01" in r.text
    assert client.get("/samples/nope").status_code == 404


def brute_force_pair_trials(ds: Dataset) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """For each open, still-matchable pair, brute-force (same_lo, same_hi,
    diff_lo, diff_hi) match counts."""
    forced = ds.verdicts.same
    blocked1 = {a for a, _ in forced}
    blocked2 = {b for _, b in forced}
    out = {}
    for c in ds.candidates:
        if c.key in forced or c.key in ds.verdicts.diff:
            continue
        if c.round1_id in blocked1 or c.round2_id in blocked2:
            continue
        same_counts, diff_counts = [], []
        for verdict, bucket in (("same", same_counts), ("diff", diff_counts)):
            trial = Dataset(ds.round1, ds.round2, ds.candidates,
                            Verdicts(set(forced), set(ds.verdicts.diff)))
            if verdict == "same":
                trial.verdicts.same.add(c.key)
            else:
                trial.verdicts.diff.add(c.key)
            lo, hi = brute_force_m_range(trial)
            bucket.extend([lo, hi])
        out[c.key] = (*same_counts, *diff_counts)
    return out


def test_pair_trials_match_brute_force():
    rng = random.Random(99)
    for _ in range(25):
        n1, n2 = rng.randint(2, 4), rng.randint(2, 4)
        all_pairs = [(a, b) for a in range(n1) for b in range(n2)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[: rng.randint(1, 7)]
        ds = make(n1, n2, pairs)
        expected = brute_force_pair_trials(ds)
        rec = recommend(ds)
        assert {t.candidate.key for t in rec.trials} == set(expected)
        for t in rec.trials:
            assert (t.same.m_min, t.same.m_max, t.diff.m_min, t.diff.m_max) == (
                expected[t.candidate.key]
            )


def test_recommendation_minimises_wider_trial_width():
    rng = random.Random(1234)
    for _ in range(20):
        n1, n2 = rng.randint(2, 5), rng.randint(2, 5)
        all_pairs = [(a, b) for a in range(n1) for b in range(n2)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[: rng.randint(2, 8)]
        ds = make(n1, n2, pairs)
        rec = recommend(ds)
        widths = [t.effective_width for t in rec.trials]
        assert rec.best
        assert rec.best_width == min(widths)
        assert all(t.effective_width == rec.best_width for t in rec.best)


def test_uploaded_files_used_for_survey(client):
    r = client.post(
        "/surveys",
        data={"name": "文件上传"},
        files={
            "round1_file": ("r1.txt", "A1\nA2\n".encode(), "text/plain"),
            "round2_file": ("r2.txt", "B1\n".encode(), "text/plain"),
            "pairs_file": ("pairs.txt", "A1,B1\nA2,B1\n".encode(), "text/plain"),
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    api = client.get(r.headers["location"].replace("/surveys/", "/api/surveys/")).json()
    assert api["estimate"]["n1"] == 2 and api["estimate"]["n2"] == 1
    assert api["recommendation"]["considered"] == 2


def test_concurrent_same_verdicts_only_one_wins():
    import threading

    import tempfile as _tempfile

    from app import db
    from app.model import Dataset as _DS

    old = os.environ.get("SEAL_DB_PATH")
    path = os.path.join(_tempfile.mkdtemp(prefix="seal-race-"), "race.db")
    os.environ["SEAL_DB_PATH"] = path
    try:
        db.init_db()
        ds = _DS.from_raw(["L0"], ["R0", "R1"], [("L0", "R0"), ("L0", "R1")])
        sid = db.create_survey(ds, "race")
        results = []

        def worker(target):
            try:
                db.apply_verdict_batch(sid, [(target[0], target[1], "same")])
                results.append(("ok", target))
            except db.ConflictError:
                results.append(("conflict", target))

        threads = [
            threading.Thread(target=worker, args=((("L0", "R0")),)),
            threading.Thread(target=worker, args=((("L0", "R1")),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(r[0] for r in results) == ["conflict", "ok"]
        final = db.load_dataset(sid)
        assert len(final.verdicts.same) == 1
    finally:
        if old is None:
            os.environ.pop("SEAL_DB_PATH", None)
        else:
            os.environ["SEAL_DB_PATH"] = old


def test_best_rows_rendered(client):
    r = client.post(
        "/surveys",
        data={
            "name": "渲染",
            "round1_text": "A1\nA2",
            "round2_text": "B1\nB2",
            "pairs_text": "A1,B1\nA2,B2",
        },
        follow_redirects=False,
    )
    page = client.get(r.headers["location"])
    assert page.status_code == 200
    assert "下一步该比对哪两张照片" in page.text
    assert "row-best" in page.text
