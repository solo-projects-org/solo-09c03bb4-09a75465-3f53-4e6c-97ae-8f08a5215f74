"""SQLite persistence for surveys, candidates and confirmed verdicts."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .model import (
    Candidate,
    Dataset,
    InputError,
    Verdicts,
    apply_verdict_changes,
)

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

CREATE INDEX IF NOT EXISTS idx_candidates_survey
    ON candidates (survey_id, position);
CREATE INDEX IF NOT EXISTS idx_verdicts_survey
    ON verdicts (survey_id);
"""


class SurveyNotFound(LookupError):
    """Raised when the requested survey id does not exist."""


class ConflictError(RuntimeError):
    """Raised when a submitted verdict batch conflicts with stored ones."""


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

def apply_verdict_batch(
    survey_id: int, changes: Sequence[tuple[str, str, str]]
) -> Verdicts:
    """Validate and persist a batch of verdicts inside one write transaction.

    The current stored conclusions are re-read under ``BEGIN IMMEDIATE`` and
    the batch is re-validated against them, so concurrent submissions cannot
    silently clobber each other.  On conflict nothing is written and
    :class:`ConflictError` is raised with a user-facing message.
    """
    with connect(immediate=True) as conn:
        ds = _load_dataset(conn, survey_id)
        if ds is None:
            raise SurveyNotFound(survey_id)
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
