#!/usr/bin/env python3
"""Self-check harness: candidate contention, verdict conflicts, bounds.

Run from the repository root:

    python3 tools/selfcheck.py

It exercises the real domain layer (no HTTP) and prints a PASS/FAIL report.
"""

from __future__ import annotations

import itertools
import random
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from app.model import (  # noqa: E402
    Dataset,
    InputError,
    Verdicts,
    apply_verdict_changes,
    chapman,
    recommend,
)

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}  {detail}")


def brute_force_m_range(ds: Dataset) -> tuple[int, int]:
    forced = ds.verdicts.same
    banned = ds.verdicts.diff
    open_pairs = [c.key for c in ds.candidates if c.key not in forced | banned]
    counts = []
    for r in range(len(open_pairs) + 1):
        for combo in itertools.combinations(open_pairs, r):
            chosen = list(forced) + list(combo)
            left = [a for a, _ in chosen]
            right = [b for _, b in chosen]
            if len(set(left)) == len(left) and len(set(right)) == len(right):
                counts.append(len(chosen))
    return min(counts), max(counts)


def random_dataset(rng: random.Random, n1: int, n2: int, max_edges: int) -> Dataset:
    pairs = [(a, b) for a in range(n1) for b in range(n2)]
    rng.shuffle(pairs)
    pairs = pairs[: rng.randint(1, max_edges)]
    return Dataset.from_raw(
        [f"L{i}" for i in range(n1)],
        [f"R{i}" for i in range(n2)],
        [(f"L{a}", f"R{b}") for a, b in pairs],
    )


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    section("1. Chapman monotonicity")
    check(chapman(10, 10, 0) == 120.0, "N(10,10,m=0)=120")
    check(chapman(10, 10, 2) == 121 / 3 - 1, "N(10,10,m=2)=121/3-1")
    check(
        all(
            chapman(20, 18, m) > chapman(20, 18, m + 1)
            for m in range(0, 17)
        ),
        "N decreases strictly with m",
    )

    section("2. Bounds equal brute-force enumeration (random graphs)")
    rng = random.Random(2026)
    ok = True
    for _ in range(60):
        ds = random_dataset(rng, rng.randint(2, 5), rng.randint(2, 5), 8)
        est = ds.estimate()
        if (est.m_min, est.m_max) != brute_force_m_range(ds):
            ok = False
            break
    check(ok, "m range matches exhaustive search over 60 random graphs")

    section("3. Candidate contention")
    # two round-1 IDs compete for the same round-2 ID
    ds = Dataset.from_raw(
        ["L0", "L1"], ["R0"], [("L0", "R0"), ("L1", "R0")]
    )
    est = ds.estimate()
    check((est.m_min, est.m_max) == (0, 1), "shared R0 caps m at 1", str(est))

    # chain: forcing L0-R0 blocks L1
    ds = Dataset.from_raw(
        ["L0", "L1"], ["R0", "R1"],
        [("L0", "R0"), ("L0", "R1"), ("L1", "R0")],
    )
    est = ds.estimate()
    check((est.m_min, est.m_max) == (0, 2), "max matching over chain is 2")
    trials = {(t.candidate.round1_id, t.candidate.round2_id): t
              for t in recommend(ds).trials}
    t00 = trials[("L0", "R0")]
    check(
        (t00.same.m_min, t00.same.m_max) == (1, 1),
        "forcing contended L0-R0 freezes m at 1",
    )
    check(
        (t00.diff.m_min, t00.diff.m_max) == (0, 2),
        "excluding L0-R0 keeps m up to 2",
    )
    best = recommend(ds).best
    check(len(best) == 2, "two pairs tie for narrowest decision width")

    # full 80x80 / 800-candidate contention: perfect matching possible
    rng = random.Random(11)
    left = [f"L{i:02d}" for i in range(80)]
    right = [f"R{i:02d}" for i in range(80)]
    edges = set()
    while len(edges) < 800:
        edges.add((rng.choice(left), rng.choice(right)))
    big = Dataset.from_raw(left, right, sorted(edges))
    est = big.estimate()
    check(est.m_max == 80, "800-candidate graph reaches m=80", str(est.as_dict()))
    check(est.n_low == chapman(80, 80, 80), "lower bound uses m=80")
    check(est.n_high == chapman(80, 80, 0), "upper bound uses m=0")

    section("4. Verdict conflict rules")
    ds = Dataset.from_raw(
        ["A1", "A2"], ["B1", "B2"],
        [("A1", "B1"), ("A1", "B2"), ("A2", "B1")],
    )
    merged, msg = apply_verdict_changes(
        ds, [("A1", "B1", "same"), ("A1", "B2", "same")]
    )
    check(merged is None and "冲突" in msg, "same+samed ID rejected")
    check(ds.verdicts.same == set() and ds.verdicts.diff == set(),
          "rejected batch leaves state untouched")

    ds2 = Dataset.from_raw(
        ["A1"], ["B1"], [("A1", "B1")], Verdicts(set(), {("A1", "B1")})
    )
    merged, msg = apply_verdict_changes(ds2, [("A1", "B1", "same")])
    check(merged is None and "两只" in msg, "relabel diff -> same rejected")
    merged, msg = apply_verdict_changes(ds2, [("A1", "B1", "diff")])
    check(merged is not None, "idempotent resubmission accepted")

    ds3 = Dataset.from_raw(
        ["A1", "A2"], ["B1"], [("A1", "B1"), ("A2", "B1")]
    )
    merged, msg = apply_verdict_changes(
        ds3, [("A1", "B1", "diff"), ("A2", "B1", "same"), ("A2", "B1", "same")]
    )
    check(merged is not None and merged.diff == {("A1", "B1")},
          "mixed batch applies both verdict kinds")

    section("5. Two concurrent 'same' verdicts on one ID (SQLite lock)")
    import tempfile
    import os

    old = os.environ.get("SEAL_DB_PATH")
    path = os.path.join(tempfile.mkdtemp(prefix="seal-selfcheck-"), "race.db")
    os.environ["SEAL_DB_PATH"] = path
    try:
        db.init_db()
        race = Dataset.from_raw(
            ["L0"], ["R0", "R1"], [("L0", "R0"), ("L0", "R1")]
        )
        sid = db.create_survey(race, "race")
        results: list[str] = []
        barrier = threading.Barrier(2)

        def worker(pair: tuple[str, str]) -> None:
            barrier.wait()
            try:
                db.apply_verdict_batch(sid, [(pair[0], pair[1], "same")])
                results.append("ok")
            except db.ConflictError:
                results.append("conflict")

        threads = [
            threading.Thread(target=worker, args=(("L0", "R0"),)),
            threading.Thread(target=worker, args=(("L0", "R1"),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check(sorted(results) == ["conflict", "ok"],
              "exactly one concurrent verdict wins", str(results))
        final = db.load_dataset(sid)
        check(final is not None and len(final.verdicts.same) == 1,
              "one same verdict persisted")
    finally:
        if old is None:
            os.environ.pop("SEAL_DB_PATH", None)
        else:
            os.environ["SEAL_DB_PATH"] = old

    section("6. Samples load and produce a recommendation")
    import json

    for name in ("small", "medium"):
        sample = json.loads(
            (ROOT / "samples" / f"{name}.json").read_text(encoding="utf-8")
        )
        ds = Dataset.from_raw(
            sample["round1"], sample["round2"],
            [tuple(p) for p in sample["pairs"]],
        )
        est = ds.estimate()
        rec = recommend(ds)
        check(
            rec.best_width is not None and est.n_low <= est.n_high,
            f"sample {name}: recommendation and ordered interval",
            f"{est.as_dict()} best={rec.best_width}",
        )

    section("7. Input limits")
    try:
        Dataset.from_raw([f"A{i}" for i in range(81)], ["B"], [("A0", "B")])
        check(False, "81 IDs rejected")
    except InputError as exc:
        check("上限" in str(exc), "81 IDs rejected")
    try:
        Dataset.from_raw(["A"], ["B"], [("A", "B")] * 3)
        check(False, "duplicate pair rejected")
    except InputError as exc:
        check("重复" in str(exc), "duplicate pair rejected")

    section("8. Original-photo packages")
    import io
    import tempfile
    import zipfile

    from app.photos import PackageError, parse_photo_package

    def jpeg(tag: str) -> bytes:
        return b"\xff\xd8\xff\xe0" + tag.encode()

    def make_zip(files, csv_text):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.csv", csv_text)
            for name, data in files.items():
                archive.writestr(name, data)
        return buf.getvalue()

    good = make_zip(
        {"A1.jpg": jpeg("a1"), "A2.jpg": jpeg("a2")},
        "编号,文件名\nA1,A1.jpg\nA2,A2.jpg\n",
    )
    photos = parse_photo_package(good, ["A1", "A2"], "第一轮")
    check(
        [p.photo_id for p in photos] == ["A1", "A2"]
        and photos[0].data == jpeg("a1"),
        "valid package parses with exact bytes",
    )

    def rejects(raw, needle, label):
        try:
            parse_photo_package(raw, ["A1", "A2"], "第一轮")
        except PackageError as exc:
            check(needle in str(exc), label, str(exc))
        else:
            check(False, label, "no error raised")

    rejects(
        make_zip({"A1.jpg": jpeg("a1")}, "A1,A1.jpg\nA2,A1.jpg\n"),
        "配给了多",
        "one file for two IDs rejected",
    )
    rejects(
        make_zip(
            {"A1.jpg": jpeg("a1"), "A2.jpg": jpeg("a2"), "x.jpg": jpeg("x")},
            "A1,A1.jpg\nA2,A2.jpg\n",
        ),
        "未在清单",
        "unlisted file rejected",
    )
    rejects(
        make_zip({"A1.jpg": jpeg("a1")}, "A1,A1.jpg\n"),
        "缺少",
        "missing round ID rejected",
    )
    rejects(
        make_zip(
            {"A1.jpg": b"PNG", "A2.jpg": jpeg("a2")},
            "A1,A1.jpg\nA2,A2.jpg\n",
        ),
        "JPEG",
        "non-JPEG rejected",
    )
    rejects(
        make_zip(
            {"A1.jpg": b"\xff\xd8\xff" + b"x" * (20 * 1024 * 1024),
             "A2.jpg": jpeg("a2")},
            "A1,A1.jpg\nA2,A2.jpg\n",
        ),
        "20 MB",
        "over 20 MB rejected",
    )

    # End-to-end storage rules with a throwaway database.
    old_db = os.environ.get("SEAL_DB_PATH")
    os.environ["SEAL_DB_PATH"] = os.path.join(
        tempfile.mkdtemp(prefix="seal-photo-selfcheck-"), "photos.db"
    )
    try:
        db.init_db()
        survey = Dataset.from_raw(
            ["A1", "A2"], ["B1", "B2"], [("A1", "B1"), ("A2", "B2")]
        )
        sid = db.create_survey(survey, "photos")
        pkg = parse_photo_package(good, ["A1", "A2"], "第一轮")
        db.save_photo_package(sid, 1, pkg)
        tokens = db.candidate_tokens(sid)
        db.apply_verdict_batch(sid, [("A1", "B1", "same")], tokens)
        check(
            db.load_dataset(sid).verdicts.same == {("A1", "B1")},
            "verdict accepted against shown originals",
        )

        changed = parse_photo_package(
            make_zip(
                {"A1.jpg": jpeg("new"), "A2.jpg": jpeg("a2")},
                "A1,A1.jpg\nA2,A2.jpg\n",
            ),
            ["A1", "A2"],
            "第一轮",
        )
        try:
            db.save_photo_package(sid, 1, changed)
        except db.PhotoConflictError as exc:
            check("A1" in str(exc), "protected original swap rejected")
        else:
            check(False, "protected original swap rejected")
        served, _ = db.photo_bytes(sid, 1, "A1")
        check(served == jpeg("a1"), "originals kept after rejected swap")

        add_round2 = parse_photo_package(
            make_zip(
                {"B1.jpg": jpeg("b1"), "B2.jpg": jpeg("b2")},
                "B1,B1.jpg\nB2,B2.jpg\n",
            ),
            ["B1", "B2"],
            "第二轮",
        )
        db.save_photo_package(sid, 2, add_round2)
        stale = dict(tokens)
        try:
            db.apply_verdict_batch(sid, [("A2", "B2", "diff")], stale)
        except db.PhotoConflictError as exc:
            check("刷新" in str(exc), "stale page rejected after originals added")
        else:
            check(False, "stale page rejected after originals added")

        fresh = db.candidate_tokens(sid)
        db.apply_verdict_batch(sid, [("A2", "B2", "diff")], fresh)
        check(
            db.load_dataset(sid).verdicts.diff == {("A2", "B2")},
            "fresh page accepted after refresh",
        )
    finally:
        if old_db is None:
            os.environ.pop("SEAL_DB_PATH", None)
        else:
            os.environ["SEAL_DB_PATH"] = old_db

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
