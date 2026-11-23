"""Parsing and validation of a per-round ZIP package of original photos.

A package must contain

* exactly one two-column CSV manifest ``编号,文件名`` (a header row is
  optional), mapping every ID of the round to exactly one JPEG, and
* the JPEG files referenced by the manifest.

Rules enforced here (all-or-nothing, the caller stores nothing on error):

* every round ID appears in the manifest exactly once;
* every manifest filename belongs to exactly one ID;
* every JPEG in the ZIP is referenced by the manifest (no stray files);
* referenced files must really be JPEG (checked by magic bytes);
* the uncompressed photo bytes total at most 20 MiB.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

MAX_PHOTO_BYTES = 20 * 1024 * 1024
# A 20 MiB photo set does not compress much; leave room for the CSV and ZIP
# bookkeeping while still bounding the request body we are willing to buffer.
MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
MAX_CSV_BYTES = 1 * 1024 * 1024
MAX_ZIP_ENTRIES = 512
# The request body is buffered before it reaches us; refuse any upload whose
# declared length is above this ceiling outright.
MAX_REQUEST_BYTES = 40 * 1024 * 1024

_JPEG_MAGIC = b"\xff\xd8\xff"
_ID_HEADERS = {"编号", "id", "photo_id", "photo id", "编号id"}
_FILE_HEADERS = {"文件名", "filename", "file", "文件", "file_name", "file name"}
# Operating-system metadata that archivers add on their own.  These are not
# photos and are ignored rather than reported as unlisted files.
_IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


class PackageError(ValueError):
    """User-facing package validation error (Chinese message)."""


@dataclass(frozen=True)
class PhotoFile:
    """One validated photo, ready to be stored."""

    photo_id: str
    filename: str
    data: bytes
    sha256: str

    @property
    def size(self) -> int:
        return len(self.data)


def _norm_entry_name(name: str) -> str:
    """Normalise a ZIP entry / manifest filename for safe, exact matching."""
    name = name.replace("\\", "/").strip()
    while name.startswith("./"):
        name = name[2:]
    if not name or name.startswith("/"):
        raise PackageError(f"文件名 {name!r} 不合法。")
    path = PurePosixPath(name)
    if any(part in ("..", "") for part in path.parts) or path.is_absolute():
        raise PackageError(f"文件名 {name!r} 不合法（不允许 .. 或绝对路径）。")
    return str(path)


def _decode_csv(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise PackageError("CSV 清单无法按 UTF-8 或 GB18030 解码。")


def _read_manifest(
    text: str, round_ids: Sequence[str]
) -> list[tuple[str, str]]:
    reader = csv.reader(io.StringIO(text, newline=""))
    known_ids = set(round_ids)
    rows: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    header_checked = False
    for lineno, raw in enumerate(reader, start=1):
        cells = [cell.strip() for cell in raw]
        if not any(cells):
            continue
        if len(cells) != 2:
            raise PackageError(
                f"CSV 清单第 {lineno} 行应为两列（编号,文件名），"
                f"实际为 {len(cells)} 列。"
            )
        photo_id, filename = cells
        if not header_checked:
            header_checked = True
            if (
                photo_id.lower() in _ID_HEADERS
                and filename.lower() in _FILE_HEADERS
            ):
                continue
        if not photo_id:
            raise PackageError(f"CSV 清单第 {lineno} 行的编号为空。")
        if photo_id not in known_ids:
            raise PackageError(
                f"CSV 清单中的编号 {photo_id} 不在本轮清单中。"
            )
        if photo_id in seen_ids:
            raise PackageError(f"CSV 清单中编号 {photo_id} 重复出现。")
        seen_ids.add(photo_id)
        norm = _norm_entry_name(filename)
        if norm in seen_files:
            raise PackageError(f"清单中文件名 {filename} 被配给了多个编号。")
        seen_files.add(norm)
        rows.append((photo_id, norm))

    if not rows:
        raise PackageError("CSV 清单为空。")
    missing = [pid for pid in round_ids if pid not in seen_ids]
    if missing:
        shown = "、".join(missing[:8])
        more = "" if len(missing) <= 8 else f" 等 {len(missing)} 个"
        raise PackageError(f"CSV 清单缺少本轮编号：{shown}{more}。")
    return rows


def _read_bounded(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int
) -> bytes:
    """Read one entry, refusing to buffer more than ``limit`` bytes.

    The declared ``file_size`` in the ZIP header is not trusted on its own:
    a deflate stream can expand well beyond it.
    """
    with archive.open(info) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise PackageError("解包后照片合计超过 20 MB 上限，整包已退回。")
    return data


def parse_photo_package(
    raw: bytes, round_ids: Sequence[str], label: str
) -> list[PhotoFile]:
    """Validate a ZIP package and return the photos of one round.

    ``round_ids`` is the authoritative ID list of that round; the manifest must
    cover it exactly.  Raises :class:`PackageError` with a user-facing message.
    """
    if not raw:
        raise PackageError(f"{label}原片包为空。")
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise PackageError(
            f"{label}原片包超过 {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB，无法接收。"
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise PackageError(f"{label}原片包不是有效的 ZIP 文件。") from None

    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > MAX_ZIP_ENTRIES:
            raise PackageError(f"{label}ZIP 内文件过多，无法核对清单。")

        entries: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise PackageError(f"{label}ZIP 含加密文件，无法读取。")
            name = _norm_entry_name(info.filename)
            if PurePosixPath(name).name in _IGNORED_NAMES or name.startswith(
                "__MACOSX/"
            ):
                continue
            if name in entries:
                raise PackageError(f"ZIP 内有重名文件 {name}。")
            entries[name] = info

        csv_names = [n for n in entries if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise PackageError(
                f"{label}ZIP 中应恰有一个 CSV 清单，实际找到 {len(csv_names)} 个。"
            )
        csv_name = csv_names[0]
        csv_info = entries[csv_name]
        if csv_info.file_size > MAX_CSV_BYTES:
            raise PackageError(f"{label}CSV 清单过大。")
        manifest = _read_manifest(
            _decode_csv(_read_bounded(archive, csv_info, MAX_CSV_BYTES)),
            round_ids,
        )

        def resolve(name: str) -> str | None:
            """Locate a manifest file, allowing one archive folder level.

            A ZIP made by archiving a folder contains ``round/photo.jpg``
            while the manifest usually says just ``photo.jpg``; resolve such
            names relative to the CSV's own folder.  An exact match always
            wins.
            """
            if name in entries:
                return name
            csv_dir = str(PurePosixPath(csv_name).parent)
            if csv_dir not in ("", "."):
                nested = f"{csv_dir}/{name}"
                if nested in entries:
                    return nested
            return None

        resolved: dict[str, str] = {}
        for _, name in manifest:
            found = resolve(name)
            if found is None:
                raise PackageError(f"ZIP 中找不到清单所列文件 {name}。")
            resolved[name] = found

        referenced = set(resolved.values())
        stray = sorted(
            name
            for name in entries
            if name != csv_name and name not in referenced
        )
        if stray:
            shown = "、".join(stray[:5])
            more = "" if len(stray) <= 5 else f" 等 {len(stray)} 个"
            raise PackageError(
                f"ZIP 中有未在清单中列出的文件：{shown}{more}。"
            )

        # Check declared sizes first so a clearly oversized package is
        # rejected before any file is decompressed.
        declared = sum(entries[name].file_size for name in resolved.values())
        if declared > MAX_PHOTO_BYTES:
            raise PackageError(
                "解包后照片合计超过 20 MB 上限，整包已退回。"
            )

        photos: list[PhotoFile] = []
        total = 0
        for photo_id, filename in manifest:
            info = entries[resolved[filename]]
            remaining = MAX_PHOTO_BYTES - total
            data = _read_bounded(archive, info, remaining)
            total += len(data)
            if not data.startswith(_JPEG_MAGIC):
                raise PackageError(f"文件 {filename} 不是 JPEG 照片。")
            photos.append(
                PhotoFile(
                    photo_id=photo_id,
                    filename=filename,
                    data=data,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    return photos


def state_token(round1_sha: str | None, round2_sha: str | None) -> str:
    """Short token describing the two originals a candidate page showed.

    The token changes whenever either photo is added or replaced, and stays
    the same while unrelated photos change.
    """
    material = f"{round1_sha or '-'}|{round2_sha or '-'}"
    return hashlib.sha256(material.encode("ascii")).hexdigest()[:20]
