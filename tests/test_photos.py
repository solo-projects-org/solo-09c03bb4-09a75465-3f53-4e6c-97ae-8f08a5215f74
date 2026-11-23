"""Tests for per-round original-photo packages and photo-bound verdicts."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from urllib.parse import unquote

import pytest

os.environ.setdefault(
    "SEAL_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="seal-photo-test-"), "seals.db"),
)


def jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + payload


def package(files: dict[str, bytes], csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.csv", csv_text)
        for name, data in files.items():
            archive.writestr(name, data)
    return buf.getvalue()


def round_package(ids, payloads: dict[str, bytes] | bytes = b"photo"):
    """Build a valid round package; ``payloads`` pins per-ID bytes."""
    if isinstance(payloads, bytes):
        payloads = {pid: payloads + pid.encode() for pid in ids}
    files = {f"{pid}.jpg": jpeg(payloads[pid]) for pid in ids}
    csv_text = "编号,文件名\n" + "".join(f"{pid},{pid}.jpg\n" for pid in ids)
    return package(files, csv_text)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SEAL_DB_PATH", str(tmp_path / "http.db"))
    from fastapi.testclient import TestClient

    from app import db
    from app.main import app

    db.init_db()
    with TestClient(app) as c:
        yield c


def create(client, r1="A1\nA2", r2="B1\nB2", pairs="A1,B1\nA2,B2"):
    r = client.post(
        "/surveys",
        data={
            "name": "原片区",
            "round1_text": r1,
            "round2_text": r2,
            "pairs_text": pairs,
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    return r.headers["location"]


def upload(client, loc, round_no, raw):
    return client.post(
        f"{loc}/photos/{round_no}",
        files={"package": ("round.zip", raw, "application/zip")},
        follow_redirects=False,
    )


def token_for(page: str, r1: str, r2: str) -> str:
    """Read the hidden token a rendered candidate row carries."""
    marker = f'data-r1="{r1}" data-r2="{r2}"'
    start = page.index(marker)
    end = page.index("</tr>", start)
    row = page[start:end]
    key = row.index('name="token__')
    quote = row.index('"', key + len('name="token__'))
    value_at = row.index('value="', quote)
    value_end = row.index('"', value_at + len('value="'))
    return row[value_at + len('value="'): value_end]


# ------------------------------------------------------------------ uploads

def test_upload_package_and_serve_photos(client):
    loc = create(client)
    r = upload(client, loc, 1, round_package(["A1", "A2"], b"first"))
    assert r.status_code == 303 and "error=" not in r.headers["location"]
    page = client.get(loc).text
    assert "2/2 个编号已绑定" in page
    assert "data-compare" in page and "thumb" in page
    data = client.get(f"{loc}/photos/1/A1")
    assert data.status_code == 200
    assert data.headers["content-type"].startswith("image/jpeg")
    assert data.content == jpeg(b"firstA1")
    version = data.headers["etag"].strip('"')
    assert client.get(f"{loc}/photos/1/A1?v={version}").status_code == 200
    assert client.get(f"{loc}/photos/1/A1?v=deadbeefdeadbeef").status_code == 409
    assert client.get(f"{loc}/photos/2/B1").status_code == 404


def test_bad_manifest_rejects_whole_package_and_keeps_originals(client):
    loc = create(client)
    good = round_package(["A1", "A2"], b"keep")
    assert "error=" not in upload(client, loc, 1, good).headers["location"]
    missing = package({"A1.jpg": jpeg(b"new")}, "编号,文件名\nA1,A1.jpg\n")
    r = upload(client, loc, 1, missing)
    assert "error=" in r.headers["location"]
    assert "缺少" in unquote(r.headers["location"])
    assert client.get(f"{loc}/photos/1/A1").content == jpeg(b"keepA1")
    assert client.get(f"{loc}/photos/1/A2").content == jpeg(b"keepA2")


def test_package_rules_rejected(client):
    loc = create(client)
    cases = {
        "duplicate id": package(
            {"a.jpg": jpeg(b"a"), "b.jpg": jpeg(b"b")},
            "A1,a.jpg\nA1,b.jpg\n",
        ),
        "unknown id": package(
            {"a.jpg": jpeg(b"a"), "b.jpg": jpeg(b"b")},
            "A1,a.jpg\nA3,b.jpg\n",
        ),
        "one file two ids": package(
            {"a.jpg": jpeg(b"a")},
            "A1,a.jpg\nA2,a.jpg\n",
        ),
        "stray file": package(
            {"a.jpg": jpeg(b"a"), "b.jpg": jpeg(b"b"), "z.jpg": jpeg(b"z")},
            "A1,a.jpg\nA2,b.jpg\n",
        ),
        "not jpeg": package(
            {"a.jpg": b"GIF89a", "b.jpg": jpeg(b"b")},
            "A1,a.jpg\nA2,b.jpg\n",
        ),
        "no csv": package({"a.jpg": jpeg(b"a")}, ""),
        "empty": b"",
        "not a zip": b"not a zip at all",
    }
    for label, raw in cases.items():
        r = upload(client, loc, 1, raw)
        assert "error=" in r.headers["location"], label
    assert client.get(f"{loc}/photos/1/A1").status_code == 404


def test_package_size_limit(client):
    loc = create(client)
    big = jpeg(b"x" * (20 * 1024 * 1024))
    raw = package(
        {"A1.jpg": big, "A2.jpg": jpeg(b"b")},
        "A1,A1.jpg\nA2,A2.jpg\n",
    )
    r = upload(client, loc, 1, raw)
    assert "error=" in r.headers["location"]
    assert "20 MB" in unquote(r.headers["location"])
    assert client.get(f"{loc}/photos/1/A1").status_code == 404


def test_zip_inside_one_folder_is_accepted(client):
    loc = create(client)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("round1/manifest.csv", "A1,A1.jpg\nA2,A2.jpg\n")
        archive.writestr("round1/A1.jpg", jpeg(b"a1"))
        archive.writestr("round1/A2.jpg", jpeg(b"a2"))
        archive.writestr("__MACOSX/round1/._A1.jpg", b"junk")
    r = upload(client, loc, 1, buf.getvalue())
    assert "error=" not in r.headers["location"]
    assert client.get(f"{loc}/photos/1/A1").content == jpeg(b"a1")


# ------------------------------------------------------ replacement protection

def test_protected_original_cannot_be_replaced(client):
    loc = create(client)
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], b"old")
    ).headers["location"]
    token = token_for(client.get(loc).text, "A1", "B1")
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "same", "token__0": token},
        follow_redirects=False,
    )
    assert "error=" not in r.headers["location"]

    # Changing A1 (which carries a stored verdict) rejects the whole package.
    r = upload(client, loc, 1, round_package(["A1", "A2"], b"brand-new"))
    assert "error=" in r.headers["location"]
    assert "A1" in unquote(r.headers["location"])
    assert client.get(f"{loc}/photos/1/A1").content == jpeg(b"oldA1")
    # A2 never carried a verdict, yet the package as a whole is rejected.
    assert client.get(f"{loc}/photos/1/A2").content == jpeg(b"oldA2")

    # Re-uploading identical bytes for the protected ID is allowed.
    r = upload(client, loc, 1, round_package(["A1", "A2"], b"old"))
    assert "error=" not in r.headers["location"]


def test_unprotected_original_can_be_replaced(client):
    loc = create(client)
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], b"v1")
    ).headers["location"]
    r = upload(client, loc, 1, round_package(["A1", "A2"], b"v2"))
    assert "error=" not in r.headers["location"]
    assert client.get(f"{loc}/photos/1/A1").content == jpeg(b"v2A1")


def test_round_protection_is_independent(client):
    loc = create(client)
    assert "error=" not in upload(
        client, loc, 2, round_package(["B1", "B2"], b"r2")
    ).headers["location"]
    token = token_for(client.get(loc).text, "A1", "B1")
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "diff", "token__0": token},
        follow_redirects=False,
    )
    assert "error=" not in r.headers["location"]
    # Round 1 had no photos at all; uploading them is not a replacement.
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], b"r1")
    ).headers["location"]
    # B1 is protected in round 2.
    r = upload(client, loc, 2, round_package(["B1", "B2"], b"new"))
    assert "error=" in r.headers["location"]
    assert client.get(f"{loc}/photos/2/B1").content == jpeg(b"r2B1")


# ------------------------------------------------------------ stale page rules

def test_selected_original_added_after_page_rejected(client):
    loc = create(client)
    old_page = client.get(loc).text  # no photos yet: all tokens are "absent"
    token = token_for(old_page, "A1", "B1")
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], b"new")
    ).headers["location"]
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "same", "token__0": token},
        follow_redirects=False,
    )
    assert "error=" in r.headers["location"]
    assert "刷新" in unquote(r.headers["location"])
    # Nothing was written.
    api = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    assert api["estimate"]["m_min"] == 0


def test_selected_original_replaced_after_page_rejected(client):
    loc = create(client)
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], {"A1": b"one", "A2": b"two"})
    ).headers["location"]
    page = client.get(loc).text
    token_a1 = token_for(page, "A1", "B1")
    # Only A2 changes; A1-B1 keeps its bytes, so the page is still valid.
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], {"A1": b"one", "A2": b"changed"})
    ).headers["location"]
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__1": "diff", "token__1": token_for(page, "A2", "B2")},
        follow_redirects=False,
    )
    assert "error=" in r.headers["location"]

    # Fresh page; replace A1 itself (no verdict on A1 yet, so the swap is
    # allowed), then the old page's A1-B1 token no longer matches.
    page = client.get(loc).text
    token_a1 = token_for(page, "A1", "B1")
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], {"A1": b"new", "A2": b"changed"})
    ).headers["location"]
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "diff", "token__0": token_a1},
        follow_redirects=False,
    )
    assert "error=" in r.headers["location"]
    assert "刷新" in unquote(r.headers["location"])
    assert client.get(loc.replace("/surveys/", "/api/surveys/")).json()[
        "estimate"
    ]["m_min"] == 0


def test_only_other_ids_changed_still_submits(client):
    loc = create(client)
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], {"A1": b"one", "A2": b"two"})
    ).headers["location"]
    page = client.get(loc).text
    token = token_for(page, "A1", "B1")
    # A2's original changes; the selected pair A1-B1 keeps its bytes.
    assert "error=" not in upload(
        client, loc, 1, round_package(["A1", "A2"], {"A1": b"one", "A2": b"three"})
    ).headers["location"]
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "same", "token__0": token},
        follow_redirects=False,
    )
    assert "error=" not in r.headers["location"]
    api = client.get(loc.replace("/surveys/", "/api/surveys/")).json()
    assert api["estimate"]["m_min"] >= 1


def test_legacy_survey_without_photos_still_reviewable(client):
    loc = create(client)
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "same"},
        follow_redirects=False,
    )
    assert "error=" not in r.headers["location"]
    assert "已保存" in unquote(r.headers["location"])


def test_old_page_token_checked_per_selected_pair(client):
    loc = create(client)
    for round_no, ids in ((1, ["A1", "A2"]), (2, ["B1", "B2"])):
        assert "error=" not in upload(
            client, loc, round_no, round_package(ids, b"v1")
        ).headers["location"]
    page = client.get(loc).text
    stale = token_for(page, "A2", "B2")
    fresh = token_for(page, "A1", "B1")
    # Replace round 2 entirely; A1-B1's round-2 side (B1) changes too.
    assert "error=" not in upload(
        client, loc, 2, round_package(["B1", "B2"], b"v2")
    ).headers["location"]
    r = client.post(
        f"{loc}/verdicts",
        data={"verdict__0": "same", "token__0": fresh},
        follow_redirects=False,
    )
    assert "error=" in r.headers["location"]
    # Legacy (token-less) submission for an all-legacy batch is different:
    # here photos exist, so a token is required and stale tokens are refused.
    assert client.get(loc.replace("/surveys/", "/api/surveys/")).json()[
        "estimate"
    ]["m_min"] == 0
    assert stale != fresh
