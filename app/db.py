"""SQLite persistence for surveys, candidates and confirmed verdicts."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .model import (
    Candidate,
    Dataset,
    Verdicts,
    apply_verdict_changes,
)
from .photos import PhotoFile, state_token

DEFAULT_DB_PATH = "/data/seals.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS surveys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    round1_ids    TEXT NOT NULL,
    round2_ids    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id   INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    round1_id   TEXT NOT NULL,
    round2_id   TEXT NOT NULL,
    UNIQUE (survey_id, round1_id, round2_id)
);

CREATE TABLE IF NOT EXISTS verdicts (
    survey_id    INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    verdict      TEXT NOT NULL CHECK (verdict IN ('same', 'diff')),
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (candidate_id)
);

-- One row per accepted per-round original-photo package (swap history).
CREATE TABLE IF NOT EXISTS photo_packages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id   INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
    round       INTEGER NOT NULL CHECK (round IN (1, 2)),
    created_at  TEXT NOT NULL,
    photo_count INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL
);

-- The current original photo bound to each (survey, round, ID).  A package
-- swap replaces the rows of that round atomically.
CREATE TABLE IF NOT EXISTS photos (
    survey_id  INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
    round      INTEGER NOT NULL CHECK (round IN (1, 2)),
    photo_id   TEXT NOT NULL,
    filename   TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    data       BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (survey_id, round, photo_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_survey
    ON candidates (survey_id, position);
CREATE INDEX IF NOT EXISTS idx_verdicts_survey
    ON verdicts (survey_id);
CREATE INDEX IF NOT EXISTS idx_photos_survey
    ON photos (survey_id, round);
"""


class SurveyNotFound(LookupError):
    """Raised when the requested survey id does not exist."""


class ConflictError(RuntimeError):
    """Raised when a submitted verdict batch conflicts with stored ones."""


class PhotoConflictError(RuntimeError):
    """Raised when a package would replace a photo protected by a verdict."""


def db_path() -> Path:
    return Path(os.environ.get("SEAL_DB_PATH", DEFAULT_DB_PATH))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """Open a connection; ``immediate`` takes the write lock up front."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path, timeout=10, isolation_level=None if immediate else ""
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        if conn.in_transaction:
            conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------------ surveys

def create_survey(ds: Dataset, name: str) -> int:
    with connect(immediate=True) as conn:
        cur = conn.execute(
            "INSERT INTO surveys (name, created_at, round1_ids, round2_ids)"
            " VALUES (?, ?, ?, ?)",
            (
                name,
                _now(),
                json.dumps(ds.round1, ensure_ascii=False),
                json.dumps(ds.round2, ensure_ascii=False),
            ),
        )
        survey_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO candidates (survey_id, position, round1_id, round2_id)"
            " VALUES (?, ?, ?, ?)",
            [
                (survey_id, i, c.round1_id, c.round2_id)
                for i, c in enumerate(ds.candidates)
            ],
        )
    return survey_id


def list_surveys() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.name, s.created_at,
                   (SELECT COUNT(*) FROM candidates c WHERE c.survey_id = s.id)
                       AS candidate_count,
                   (SELECT COUNT(*) FROM verdicts v WHERE v.survey_id = s.id)
                       AS verdict_count
            FROM surveys s
            ORDER BY s.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_survey_meta(survey_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, round1_ids, round2_ids FROM surveys"
            " WHERE id = ?",
            (survey_id,),
        ).fetchone()
        if row is None:
            return None
        counts = conn.execute(
            "SELECT COUNT(*) AS total,"
            " COALESCE(SUM(CASE WHEN v.verdict = 'same' THEN 1 ELSE 0 END), 0)"
            "   AS same,"
            " COALESCE(SUM(CASE WHEN v.verdict = 'diff' THEN 1 ELSE 0 END), 0)"
            "   AS diff"
            " FROM candidates c LEFT JOIN verdicts v ON v.candidate_id = c.id"
            " WHERE c.survey_id = ?",
            (survey_id,),
        ).fetchone()
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "round1": json.loads(row["round1_ids"]),
        "round2": json.loads(row["round2_ids"]),
        "candidate_count": counts["total"] or 0,
        "same_count": counts["same"] or 0,
        "diff_count": counts["diff"] or 0,
    }


def load_dataset(survey_id: int) -> Dataset | None:
    with connect() as conn:
        return _load_dataset(conn, survey_id)


def _load_verdicts(conn: sqlite3.Connection, survey_id: int) -> Verdicts:
    verdicts = Verdicts()
    rows = conn.execute(
        "SELECT c.round1_id, c.round2_id, v.verdict"
        " FROM verdicts v JOIN candidates c ON c.id = v.candidate_id"
        " WHERE v.survey_id = ?",
        (survey_id,),
    ).fetchall()
    for row in rows:
        key = (row["round1_id"], row["round2_id"])
        if row["verdict"] == "same":
            verdicts.same.add(key)
        else:
            verdicts.diff.add(key)
    return verdicts


def _load_dataset(
    conn: sqlite3.Connection, survey_id: int
) -> Dataset | None:
    row = conn.execute(
        "SELECT id, round1_ids, round2_ids FROM surveys WHERE id = ?",
        (survey_id,),
    ).fetchone()
    if row is None:
        return None
    rows = conn.execute(
        "SELECT round1_id, round2_id FROM candidates"
        " WHERE survey_id = ? ORDER BY position",
        (survey_id,),
    ).fetchall()
    return Dataset(
        round1=json.loads(row["round1_ids"]),
        round2=json.loads(row["round2_ids"]),
        candidates=[Candidate(r["round1_id"], r["round2_id"]) for r in rows],
        verdicts=_load_verdicts(conn, survey_id),
    )


# ----------------------------------------------------------------- verdicts

def candidate_token(
    photos: Mapping[tuple[int, str], dict], a: str, b: str
) -> str:
    """Token of the two originals currently bound to a candidate pair."""
    first = photos.get((1, a))
    second = photos.get((2, b))
    return state_token(
        first["sha256"] if first else None,
        second["sha256"] if second else None,
    )


def candidate_tokens(survey_id: int) -> dict[tuple[str, str], str]:
    """Tokens for every candidate pair, from the currently stored photos."""
    with connect() as conn:
        photos = _photo_index(conn, survey_id)
        rows = conn.execute(
            "SELECT round1_id, round2_id FROM candidates WHERE survey_id = ?",
            (survey_id,),
        ).fetchall()
    return {
        (row["round1_id"], row["round2_id"]): candidate_token(
            photos, row["round1_id"], row["round2_id"]
        )
        for row in rows
    }


def apply_verdict_batch(
    survey_id: int,
    changes: Sequence[tuple[str, str, str]],
    photo_tokens: Mapping[tuple[str, str], str] | None = None,
) -> Verdicts:
    """Validate and persist a batch of verdicts inside one write transaction.

    The current stored conclusions are re-read under ``BEGIN IMMEDIATE`` and
    the batch is re-validated against them, so concurrent submissions cannot
    silently clobber each other.  On conflict nothing is written and
    :class:`ConflictError` is raised with a user-facing message.

    ``photo_tokens`` maps every reviewed candidate to the original-photo state
    the page displayed.  They are checked in the same transaction against the
    currently stored photos; if any reviewed candidate's originals were added
    or replaced since the page was rendered the whole batch is rejected and
    nothing is written.
    """
    with connect(immediate=True) as conn:
        ds = _load_dataset(conn, survey_id)
        if ds is None:
            raise SurveyNotFound(survey_id)
        if photo_tokens is not None:
            photos = _photo_index(conn, survey_id)
            stale = [
                (a, b)
                for a, b, _ in changes
                if photo_tokens.get((a, b)) != candidate_token(photos, a, b)
            ]
            if stale:
                shown = "、".join(f"{a}-{b}" for a, b in sorted(stale)[:5])
                more = "" if len(stale) <= 5 else f" 等 {len(stale)} 对"
                raise PhotoConflictError(
                    f"页面所依据的原片已增加或更换（{shown}{more}），"
                    "本批已整批退回，请刷新后按当前原片重新核对。"
                )
        merged, message = apply_verdict_changes(ds, changes)
        if merged is None:
            raise ConflictError(message)
        ids = {
            (row["round1_id"], row["round2_id"]): row["id"]
            for row in conn.execute(
                "SELECT id, round1_id, round2_id FROM candidates"
                " WHERE survey_id = ?",
                (survey_id,),
            ).fetchall()
        }
        now = _now()
        conn.execute("DELETE FROM verdicts WHERE survey_id = ?", (survey_id,))
        conn.executemany(
            "INSERT INTO verdicts (survey_id, candidate_id, verdict, updated_at)"
            " VALUES (?, ?, ?, ?)",
            [(survey_id, ids[key], "same", now) for key in sorted(merged.same)]
            + [
                (survey_id, ids[key], "diff", now)
                for key in sorted(merged.diff)
            ],
        )
        return merged


def verdict_rows(survey_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.round1_id, c.round2_id, v.verdict, v.updated_at"
            " FROM verdicts v JOIN candidates c ON c.id = v.candidate_id"
            " WHERE v.survey_id = ? ORDER BY v.updated_at, c.position",
            (survey_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ------------------------------------------------------------------- photos

def _photo_index(
    conn: sqlite3.Connection, survey_id: int
) -> dict[tuple[int, str], dict]:
    """Current photo metadata keyed by (round, photo_id), without blobs."""
    rows = conn.execute(
        "SELECT round, photo_id, filename, sha256, size, updated_at"
        " FROM photos WHERE survey_id = ?",
        (survey_id,),
    ).fetchall()
    return {
        (row["round"], row["photo_id"]): {
            "filename": row["filename"],
            "sha256": row["sha256"],
            "size": row["size"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    }


def photo_index(survey_id: int) -> dict[tuple[int, str], dict]:
    with connect() as conn:
        return _photo_index(conn, survey_id)


def protected_ids(
    conn: sqlite3.Connection, survey_id: int
) -> tuple[set[str], set[str]]:
    """Round-1 / round-2 IDs that already carry a stored verdict.

    Such IDs are protected only when a photo is actually bound to them; the
    package swap re-checks that with the current photo index.
    """
    rows = conn.execute(
        "SELECT c.round1_id, c.round2_id FROM verdicts v"
        " JOIN candidates c ON c.id = v.candidate_id"
        " WHERE v.survey_id = ?",
        (survey_id,),
    ).fetchall()
    return (
        {row["round1_id"] for row in rows},
        {row["round2_id"] for row in rows},
    )


def save_photo_package(
    survey_id: int, round_no: int, photos: Sequence[PhotoFile]
) -> dict:
    """Atomically install a round's originals, enforcing the swap rules.

    * IDs whose photo is already bound *and* that carry a stored verdict are
      protected: their bytes may not change.  Any violation rejects the whole
      package and leaves the stored originals untouched.
    * A successful swap replaces all photos of that round in one transaction,
      so package swaps and verdict submissions serialise in commit order.
      Verdicts are *never* silently invalidated: their rows keep pointing at
      existing photos, and the verdict batch is what refuses a stale page.
    """
    if round_no not in (1, 2):
        raise ValueError("round must be 1 or 2")
    with connect(immediate=True) as conn:
        survey = conn.execute(
            "SELECT round1_ids, round2_ids FROM surveys WHERE id = ?",
            (survey_id,),
        ).fetchone()
        if survey is None:
            raise SurveyNotFound(survey_id)
        expected = set(
            json.loads(
                survey["round1_ids"] if round_no == 1 else survey["round2_ids"]
            )
        )
        incoming_ids = {photo.photo_id for photo in photos}
        if incoming_ids != expected or len(incoming_ids) != len(photos):
            missing = sorted(expected - incoming_ids)
            extra = sorted(incoming_ids - expected)
            detail = []
            if missing:
                detail.append("缺少 " + "、".join(missing[:6]))
            if extra:
                detail.append("多出 " + "、".join(extra[:6]))
            raise PhotoConflictError(
                "清单与本轮编号不符（" + "；".join(detail) + "），整包已退回。"
            )

        current = _photo_index(conn, survey_id)
        protected1, protected2 = protected_ids(conn, survey_id)
        protected = protected1 if round_no == 1 else protected2
        changed: list[str] = []
        for photo in photos:
            old = current.get((round_no, photo.photo_id))
            if old is None or photo.photo_id not in protected:
                continue
            if old["sha256"] != photo.sha256:
                changed.append(photo.photo_id)
        if changed:
            shown = "、".join(sorted(changed)[:8])
            more = "" if len(changed) <= 8 else f" 等 {len(changed)} 个"
            raise PhotoConflictError(
                f"已核对编号 {shown}{more} 的原片被更换；这些编号的核对结论"
                "以原片为依据，整包已退回并保留原片。"
            )

        return _install_photos(conn, survey_id, round_no, photos, current)


def _install_photos(
    conn: sqlite3.Connection,
    survey_id: int,
    round_no: int,
    photos: Sequence[PhotoFile],
    current: Mapping[tuple[int, str], dict],
) -> dict:
    now = _now()
    conn.execute(
        "DELETE FROM photos WHERE survey_id = ? AND round = ?",
        (survey_id, round_no),
    )
    conn.executemany(
        "INSERT INTO photos (survey_id, round, photo_id, filename,"
        " sha256, size, data, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                survey_id,
                round_no,
                photo.photo_id,
                photo.filename,
                photo.sha256,
                photo.size,
                sqlite3.Binary(photo.data),
                now,
            )
            for photo in photos
        ],
    )
    total = sum(photo.size for photo in photos)
    cur = conn.execute(
        "INSERT INTO photo_packages (survey_id, round, created_at,"
        " photo_count, total_bytes) VALUES (?, ?, ?, ?, ?)",
        (survey_id, round_no, now, len(photos), total),
    )
    replaced = any(
        (round_no, photo.photo_id) in current for photo in photos
    )
    return {
        "package_id": int(cur.lastrowid),
        "round": round_no,
        "photo_count": len(photos),
        "total_bytes": total,
        "replaced": replaced,
        "changed_protected": [],
    }


def photo_bytes(
    survey_id: int, round_no: int, photo_id: str
) -> tuple[bytes, dict] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT data, filename, sha256, size, updated_at FROM photos"
            " WHERE survey_id = ? AND round = ? AND photo_id = ?",
            (survey_id, round_no, photo_id),
        ).fetchone()
    if row is None:
        return None
    return bytes(row["data"]), {
        "filename": row["filename"],
        "sha256": row["sha256"],
        "size": row["size"],
        "updated_at": row["updated_at"],
    }


def package_rows(survey_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT round, created_at, photo_count, total_bytes"
            " FROM photo_packages WHERE survey_id = ?"
            " ORDER BY id DESC",
            (survey_id,),
        ).fetchall()
    return [dict(row) for row in rows]
