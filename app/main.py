"""FastAPI application for reviewing cross-round seal photo matches."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from .model import (
    MAX_CANDIDATES,
    MAX_IDS_PER_ROUND,
    Dataset,
    InputError,
    parse_ids_text,
    parse_pairs_text,
)
from .photos import (
    MAX_ARCHIVE_BYTES,
    MAX_REQUEST_BYTES,
    PackageError,
    PhotoFile,
    parse_photo_package,
    state_token,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR.parent / "samples"
MAX_PHOTO_MB = 20


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="海豹照片重捕核对", lifespan=lifespan)
app.mount(
    "/static", StaticFiles(directory=BASE_DIR / "static"), name="static"
)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _get_dataset_or_404(survey_id: int) -> Dataset:
    ds = db.load_dataset(survey_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="调查记录不存在")
    return ds


def _analysis(ds: Dataset) -> dict:
    est = ds.estimate()
    rec = ds.recommendation()
    return {
        "estimate": est,
        "recommendation": rec,
        "open_count": len(ds.open_candidates()),
        "pair_trials": rec.trials,
    }


@app.middleware("http")
async def limit_upload_body(request: Request, call_next):
    """Reject obviously oversized uploads before buffering the whole body."""
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            if request.url.path.startswith("/surveys/") and "/photos/" in request.url.path:
                survey_id = request.url.path.split("/")[2]
                if survey_id.isdigit():
                    return _photo_redirect(
                        int(survey_id),
                        error="上传内容过大：原片包上限为 24 MB，解包后照片合计最多 20 MB。",
                    )
            return Response("上传内容过大。", status_code=413)
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "surveys": db.list_surveys(),
            "max_ids": MAX_IDS_PER_ROUND,
            "max_pairs": MAX_CANDIDATES,
            "error": request.query_params.get("error", ""),
            "notice": request.query_params.get("notice", ""),
        },
    )


@app.post("/surveys")
async def create_survey(
    request: Request,
    name: str = Form(""),
    round1_text: str = Form(""),
    round2_text: str = Form(""),
    pairs_text: str = Form(""),
    round1_file: UploadFile | None = File(None),
    round2_file: UploadFile | None = File(None),
    pairs_file: UploadFile | None = File(None),
):
    try:
        round1_text = await _combined_text(round1_text, round1_file, "第一轮")
        round2_text = await _combined_text(round2_text, round2_file, "第二轮")
        pairs_text = await _combined_text(pairs_text, pairs_file, "疑似对")
        if len(pairs_text) > 2_000_000:
            raise InputError("粘贴内容过长。")
        r1 = parse_ids_text(round1_text, "第一轮")
        r2 = parse_ids_text(round2_text, "第二轮")
        pairs = parse_pairs_text(pairs_text) if pairs_text.strip() else []
        if not pairs:
            raise InputError("跨轮疑似同只的编号对为空。")
        ds = Dataset.from_raw(r1, r2, pairs)
        survey_name = name.strip() or "未命名调查"
        survey_id = db.create_survey(ds, survey_name)
    except InputError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "surveys": db.list_surveys(),
                "max_ids": MAX_IDS_PER_ROUND,
                "max_pairs": MAX_CANDIDATES,
                "error": str(exc),
                "notice": "",
                "form": {
                    "name": name,
                    "round1_text": round1_text,
                    "round2_text": round2_text,
                    "pairs_text": pairs_text,
                },
            },
            status_code=400,
        )
    return RedirectResponse(f"/surveys/{survey_id}", status_code=303)


@app.get("/surveys/{survey_id}", response_class=HTMLResponse)
def survey_detail(request: Request, survey_id: int):
    ds = _get_dataset_or_404(survey_id)
    analysis = _analysis(ds)
    meta = db.get_survey_meta(survey_id)
    photos = db.photo_index(survey_id)
    tokens = {
        c.key: db.candidate_token(photos, c.round1_id, c.round2_id)
        for c in ds.candidates
    }
    return templates.TemplateResponse(
        request,
        "survey.html",
        {
            "survey": meta,
            "dataset": ds,
            "analysis": analysis,
            "verdict_rows": db.verdict_rows(survey_id),
            "photo1": {
                pid: meta_row
                for (rnd, pid), meta_row in photos.items()
                if rnd == 1
            },
            "photo2": {
                pid: meta_row
                for (rnd, pid), meta_row in photos.items()
                if rnd == 2
            },
            "photo_tokens": tokens,
            "packages": db.package_rows(survey_id),
            "max_photo_mb": MAX_PHOTO_MB,
            "error": request.query_params.get("error", ""),
            "notice": request.query_params.get("notice", ""),
        },
    )


@app.get("/surveys/{survey_id}/photos/{round_no}")
def photo_round_page(survey_id: int, round_no: int):
    """Convenience: the package upload form lives on the survey page."""
    return RedirectResponse(f"/surveys/{survey_id}#photo-packages", status_code=303)


@app.api_route(
    "/surveys/{survey_id}/photos/{round_no}/{photo_id:path}",
    methods=["GET", "HEAD"],
)
def survey_photo(
    survey_id: int, round_no: int, photo_id: str, v: str = ""
):
    """Serve one stored original; ``v`` pins the byte version the page saw."""
    if round_no not in (1, 2):
        raise HTTPException(status_code=404, detail="轮次不存在")
    _get_dataset_or_404(survey_id)
    found = db.photo_bytes(survey_id, round_no, photo_id)
    if found is None:
        raise HTTPException(status_code=404, detail="原片不存在")
    data, meta = found
    if v and v != meta["sha256"][:16]:
        # The page was rendered against an older original; make the stale
        # reference fail loudly instead of silently showing another photo.
        raise HTTPException(
            status_code=409, detail="原片已更换，请刷新页面"
        )
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{meta["sha256"][:16]}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/surveys/{survey_id}/photos/{round_no}")
async def upload_photo_package(
    survey_id: int,
    round_no: int,
    package: UploadFile | None = File(None),
):
    ds = db.load_dataset(survey_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="调查记录不存在")
    if round_no not in (1, 2):
        raise HTTPException(status_code=404, detail="轮次不存在")
    label = "第一轮" if round_no == 1 else "第二轮"
    round_ids = ds.round1 if round_no == 1 else ds.round2
    if package is None or not package.filename:
        return _photo_redirect(survey_id, error=f"请选择{label}原片 ZIP 包。")
    try:
        raw = await package.read(MAX_ARCHIVE_BYTES + 1)
        if len(raw) > MAX_ARCHIVE_BYTES:
            raise PackageError(
                f"{label}原片包超过 {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB，"
                "整包已退回。"
            )
        photos: list[PhotoFile] = parse_photo_package(raw, round_ids, label)
        result = db.save_photo_package(survey_id, round_no, photos)
    except PackageError as exc:
        return _photo_redirect(survey_id, error=str(exc))
    except db.PhotoConflictError as exc:
        return _photo_redirect(survey_id, error=str(exc))
    except db.SurveyNotFound:
        raise HTTPException(status_code=404, detail="调查记录不存在")
    action = "已换包" if result["replaced"] else "已上传"
    return _photo_redirect(
        survey_id,
        notice=(
            f"{label}原片{action}：{result['photo_count']} 张，合计 "
            f"{result['total_bytes'] / 1024 / 1024:.2f} MB。"
        ),
    )


def _photo_redirect(
    survey_id: int, *, error: str = "", notice: str = ""
) -> RedirectResponse:
    query = f"error={_quote(error)}" if error else f"notice={_quote(notice)}"
    return RedirectResponse(
        f"/surveys/{survey_id}?{query}", status_code=303
    )


@app.post("/surveys/{survey_id}/verdicts")
async def submit_verdicts(request: Request, survey_id: int):
    ds = _get_dataset_or_404(survey_id)
    form = await request.form()
    # Once originals exist, every verdict must be grounded in the page's
    # photos.  A page rendered before any photos existed carries no token; it
    # counts as "no original was shown", so it may still be submitted unless
    # the photos of a reviewed ID were added or replaced in the meantime.
    changes: list[tuple[str, str, str]] = []
    tokens: dict[tuple[str, str], str] = {}
    missing_token = state_token(None, None)
    # Form fields are named verdict__<index> with value same|diff|open, and
    # token__<index> carries the original-photo version shown on the page.
    for key, value in form.items():
        if not key.startswith("verdict__"):
            continue
        if value not in ("same", "diff"):
            continue
        try:
            idx = int(key.removeprefix("verdict__"))
        except ValueError:
            continue
        if 0 <= idx < len(ds.candidates):
            c = ds.candidates[idx]
            changes.append((c.round1_id, c.round2_id, str(value)))
            token = form.get(f"token__{idx}")
            tokens[c.key] = (
                token if isinstance(token, str) and token else missing_token
            )

    if not changes:
        return RedirectResponse(
            f"/surveys/{survey_id}?notice={_quote('未选择任何核对结论')}",
            status_code=303,
        )

    try:
        db.apply_verdict_batch(survey_id, changes, tokens)
    except db.PhotoConflictError as exc:
        return RedirectResponse(
            f"/surveys/{survey_id}?error={_quote(str(exc))}", status_code=303
        )
    except db.ConflictError as exc:
        # Conflict: nothing was written; stored conclusions and the results
        # derived from them stay exactly as they were.
        return RedirectResponse(
            f"/surveys/{survey_id}?error={_quote(str(exc))}", status_code=303
        )
    except db.SurveyNotFound:
        raise HTTPException(status_code=404, detail="调查记录不存在")
    return RedirectResponse(
        f"/surveys/{survey_id}?notice={_quote('核对结论已保存，估计与推荐已刷新')}",
        status_code=303,
    )


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


MAX_UPLOAD_BYTES = 1_000_000


async def _combined_text(
    pasted: str, upload: UploadFile | None, label: str
) -> str:
    """Use the uploaded text file when provided, otherwise the pasted text."""
    if upload is None or not upload.filename:
        return pasted
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(f"{label}文件超过 {MAX_UPLOAD_BYTES // 1000} KB。")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise InputError(f"{label}文件无法按 UTF-8 或 GB18030 解码。")
    return text


@app.get("/api/surveys/{survey_id}")
def api_survey(survey_id: int):
    ds = _get_dataset_or_404(survey_id)
    est = ds.estimate()
    rec = ds.recommendation()
    return {
        "survey_id": survey_id,
        "estimate": est.as_dict(),
        "recommendation": {
            "considered": rec.considered,
            "best": [t.as_dict() for t in rec.best],
            "trials": [t.as_dict() for t in rec.trials],
        },
    }


@app.get("/samples/{sample_name}", response_class=HTMLResponse)
def load_sample(request: Request, sample_name: str):
    """Load a bundled example into the create form (prefilled textareas)."""
    path = (SAMPLE_DIR / f"{sample_name}.json").resolve()
    if SAMPLE_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="样例不存在")
    sample = json.loads(path.read_text(encoding="utf-8"))
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "surveys": db.list_surveys(),
            "max_ids": MAX_IDS_PER_ROUND,
            "max_pairs": MAX_CANDIDATES,
            "error": "",
            "notice": f"已载入样例「{sample['name']}」，可直接提交。",
            "form": {
                "name": sample["name"],
                "round1_text": "\n".join(sample["round1"]),
                "round2_text": "\n".join(sample["round2"]),
                "pairs_text": "\n".join(
                    f"{a},{b}" for a, b in sample["pairs"]
                ),
            },
        },
    )
