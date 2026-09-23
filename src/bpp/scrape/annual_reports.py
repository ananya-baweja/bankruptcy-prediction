"""Collect annual report PDFs for every firm-year in the sample frame.

Three ways a report gets into the project (checked in this order):

1. **Manual** - someone downloads the PDF and either saves it as
   ``data/raw/annual_reports/<firm_id>/FY<year>.pdf`` or lists it in
   ``data/manual/manual_reports.csv``. Manual always wins. Use this for
   suspended/delisted firms whose reports are no longer on the exchange sites
   (company websites, MCA filings, Wayback Machine).
2. **NSE** - via the open-source ``nse`` package (github.com/BennyThadikaran/NseIndiaApi),
   which handles NSE's cookies. ``NSE.annual_reports(symbol)`` returns one entry
   per year with a PDF link.
3. **BSE** - the BSE annual-report API. The endpoint in config is UNVERIFIED
   (BSE returns 403 without browser headers and changes endpoints). Test with one
   scrip code first: ``bpp reports-list --firm-id BSE500325 --source bse``.

Every file ends up in the manifest ``data/interim/documents.csv``:
doc_id, firm_id, fy, source, url, local_path, pub_date, sha256, n_bytes, status
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bpp.common import PoliteSession, doc_id_for, parse_fy_label, to_timestamp, write_csv
from bpp.config import PROJECT_ROOT, Paths

log = logging.getLogger(__name__)

MANIFEST_COLS = ["doc_id", "firm_id", "fy", "source", "url", "local_path", "pub_date",
                 "sha256", "n_bytes", "status", "note"]


# ----------------------------------------------------------------------------- manifest
def load_manifest(paths: Paths) -> pd.DataFrame:
    if paths.documents.exists():
        return pd.read_csv(paths.documents, dtype=str)
    return pd.DataFrame(columns=MANIFEST_COLS)


def upsert_manifest(paths: Paths, records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    man = load_manifest(paths)
    new = pd.DataFrame(list(records), columns=MANIFEST_COLS).astype(str).replace({"None": "", "nan": ""})
    if new.empty:
        return man
    man = pd.concat([man[~man["doc_id"].isin(new["doc_id"])], new], ignore_index=True)
    man = man.sort_values("doc_id").reset_index(drop=True)
    write_csv(man, paths.documents)
    return man


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_path(paths: Paths, firm_id: str, fy: int) -> Path:
    return paths.raw_reports / firm_id / f"FY{int(fy)}.pdf"


def save_pdf_bytes(content: bytes, dest: Path) -> bool:
    """Save PDF bytes; unzip if the exchange served a zip. Returns False if not a PDF."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            pdfs = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
            if not pdfs:
                return False
            content = zf.read(max(pdfs, key=lambda n: zf.getinfo(n).file_size))
    if not content.lstrip()[:5].startswith(b"%PDF"):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return True


# ----------------------------------------------------------------------------- manual
def register_manual_reports(paths: Paths, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Register PDFs placed by hand (naming convention + manual_reports.csv)."""
    records = []
    wanted = None
    if frame is not None:
        wanted = {doc_id_for(f, y) for f, y in zip(frame["firm_id"], frame["fy"])}

    existing = load_manifest(paths)
    known_pub = dict(zip(existing["doc_id"], existing["pub_date"])) if len(existing) else {}
    known_src = dict(zip(existing["doc_id"], existing["source"])) if len(existing) else {}
    known_url = dict(zip(existing["doc_id"], existing["url"])) if len(existing) else {}

    # (a) naming convention data/raw/annual_reports/<firm_id>/FY<year>.pdf
    for pdf in sorted(paths.raw_reports.glob("*/FY*.pdf")):
        fy = parse_fy_label(pdf.stem)
        firm_id = pdf.parent.name
        if fy is None or firm_id.startswith("_"):
            continue
        did = doc_id_for(firm_id, fy)
        if wanted is not None and did not in wanted:
            continue
        records.append({"doc_id": did, "firm_id": firm_id, "fy": fy,
                        "source": known_src.get(did) or "manual", "url": known_url.get(did, ""),
                        "local_path": _rel(pdf), "pub_date": known_pub.get(did, ""),
                        "sha256": _sha256(pdf), "n_bytes": pdf.stat().st_size,
                        "status": "registered", "note": ""})

    # (b) manual_reports.csv (can point anywhere and carries publication dates)
    if paths.manual_reports.exists():
        m = pd.read_csv(paths.manual_reports, dtype=str).fillna("")
        by_id = {r["doc_id"]: r for r in records}
        for _, r in m.iterrows():
            fy = parse_fy_label(r.get("fy"))
            if not r.get("firm_id") or fy is None:
                continue
            did = doc_id_for(r["firm_id"], fy)
            p = Path(r.get("local_path", "")) if r.get("local_path") else target_path(paths, r["firm_id"], fy)
            if not p.is_absolute():
                p = (paths.data / p) if (paths.data / p).exists() else (PROJECT_ROOT / p)
            if not p.exists():
                log.warning("manual_reports.csv: %s not found for %s", p, did)
                continue
            pub = to_timestamp(r.get("pub_date"))
            rec = by_id.get(did) or {"doc_id": did, "firm_id": r["firm_id"], "fy": fy}
            rec.update({"source": "manual", "url": r.get("source_url", ""), "local_path": _rel(p),
                        "pub_date": pub.date().isoformat() if pub is not None else rec.get("pub_date", ""),
                        "sha256": _sha256(p), "n_bytes": p.stat().st_size, "status": "registered",
                        "note": r.get("note", "")})
            by_id[did] = rec
        records = list(by_id.values())

    man = upsert_manifest(paths, records)
    log.info("registered %d local PDFs; manifest now has %d documents", len(records), len(man))
    return man


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def resolve_local_path(local_path: str) -> Path:
    p = Path(local_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ----------------------------------------------------------------------------- listings
def parse_nse_listing(payload: Any) -> list[dict[str, Any]]:
    """NSE annual_reports() -> [{fy, url, pub_date}]. Field names parsed defensively."""
    items = payload.get("data", []) if isinstance(payload, dict) else (payload or [])
    out = []
    for it in items:
        low = {k.lower(): v for k, v in it.items()}
        url = low.get("filename") or next((v for v in it.values()
                                           if isinstance(v, str) and re.search(r"\.(pdf|zip)$", v, re.I)), None)
        fy = parse_fy_label(low.get("toyr")) or parse_fy_label(f"{low.get('fromyr')}-{low.get('toyr')}")
        if fy is None and url:
            fy = parse_fy_label(url)
        date_val = next((v for k, v in low.items() if ("date" in k or "dttm" in k or k.endswith("dt")) and v), None)
        pub = to_timestamp(date_val)
        if url and fy:
            out.append({"fy": fy, "url": url, "pub_date": pub.date().isoformat() if pub is not None else ""})
    return out


def parse_bse_listing(payload: Any, scripcode: str, pdf_base: str) -> list[dict[str, Any]]:
    """BSE annual report JSON -> [{fy, url, pub_date}]. Field names parsed defensively."""
    items = payload if isinstance(payload, list) else (payload or {}).get("Table", [])
    out = []
    for it in items:
        low = {k.lower(): v for k, v in it.items()}
        fname = next((v for v in it.values() if isinstance(v, str) and v.lower().endswith(".pdf")), None)
        if not fname:
            continue
        url = fname if fname.startswith("http") else f"{pdf_base.rstrip('/')}/{scripcode}/{fname}"
        year_val = next((v for k, v in low.items() if "year" in k and v), None)
        fy = parse_fy_label(year_val) or parse_fy_label(fname)
        date_val = next((v for k, v in low.items() if ("date" in k or "dt" in k) and v), None)
        pub = to_timestamp(date_val)
        if fy:
            out.append({"fy": fy, "url": url, "pub_date": pub.date().isoformat() if pub is not None else ""})
    return out


class NseClient:
    """Thin wrapper around the ``nse`` package (optional dependency)."""

    def __init__(self, paths: Paths):
        try:
            from nse import NSE  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install nse   (needed for NSE annual reports)") from exc
        cookie_dir = paths.report_listings / "nse_cookies"
        cookie_dir.mkdir(parents=True, exist_ok=True)
        self.nse = NSE(download_folder=cookie_dir)

    def listing(self, symbol: str) -> Any:
        return self.nse.annual_reports(symbol=symbol)

    def download(self, url: str, dest: Path) -> bool:
        tmp_dir = dest.parent / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        got = Path(self.nse.download_document(url, folder=tmp_dir))
        files = [got] if got.is_file() else list(got.glob("**/*.pdf"))
        ok = False
        if files:
            ok = save_pdf_bytes(files[0].read_bytes(), dest)
        for f in tmp_dir.glob("**/*"):
            if f.is_file():
                f.unlink()
        return ok

    def close(self) -> None:
        self.nse.exit()


class BseClient:
    def __init__(self, cfg: dict[str, Any]):
        rcfg = cfg["reports"]
        self.cfg = rcfg
        self.api = cfg["listed"]["bse_api"]
        self.sess = PoliteSession(delay_s=rcfg["request_delay_s"], timeout_s=rcfg["timeout_s"],
                                  headers={"User-Agent": rcfg["user_agent"],
                                           "Referer": "https://www.bseindia.com/",
                                           "Origin": "https://www.bseindia.com",
                                           "Accept": "application/json, text/plain, */*"})

    def listing(self, scripcode: str) -> Any:
        url = f"{self.api}/{self.cfg['bse_annual_report_endpoint']}"
        return self.sess.get(url, params={"scripcode": scripcode}).json()

    def download(self, url: str, dest: Path) -> bool:
        return save_pdf_bytes(self.sess.get(url).content, dest)

    def close(self) -> None:
        pass


def list_available_reports(cfg: dict[str, Any], paths: Paths, frame: pd.DataFrame,
                           universe: pd.DataFrame, sources: list[str],
                           firm_ids: list[str] | None = None) -> pd.DataFrame:
    """Ask NSE/BSE which annual reports exist for each cohort firm (cached as JSON)."""
    firms = frame[["firm_id"]].drop_duplicates()
    if firm_ids:
        firms = firms[firms["firm_id"].isin(firm_ids)]
    firms = firms.merge(universe[["firm_id", "bse_code", "nse_symbol"]], on="firm_id", how="left")
    rows = []
    clients: dict[str, Any] = {}
    try:
        for _, f in firms.iterrows():
            for src in sources:
                key = f["nse_symbol"] if src == "nse" else f["bse_code"]
                if pd.isna(key) or not str(key).strip():
                    continue
                key = re.sub(r"\.0$", "", str(key).strip())
                cache = paths.report_listings / f"{src}_{key}.json"
                try:
                    if cache.exists():
                        payload = json.loads(cache.read_text(encoding="utf-8"))
                    else:
                        if src not in clients:
                            clients[src] = NseClient(paths) if src == "nse" else BseClient(cfg)
                        payload = clients[src].listing(key)
                        cache.write_text(json.dumps(payload, default=str), encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s listing failed for %s (%s): %s", src, f["firm_id"], key, exc)
                    continue
                items = (parse_nse_listing(payload) if src == "nse"
                         else parse_bse_listing(payload, key, cfg["reports"]["bse_pdf_base"]))
                for it in items:
                    rows.append({"firm_id": f["firm_id"], "source": src, **it})
    finally:
        for c in clients.values():
            c.close()
    listing = pd.DataFrame(rows, columns=["firm_id", "source", "fy", "url", "pub_date"])
    write_csv(listing, paths.report_listing_table)
    return listing


def download_reports(cfg: dict[str, Any], paths: Paths, frame: pd.DataFrame,
                     listing: pd.DataFrame, sources: list[str], limit: int | None = None) -> pd.DataFrame:
    """Download the PDFs the sample frame needs, skipping anything already on disk."""
    register_manual_reports(paths, frame)            # manual files first: never overwrite them
    man = load_manifest(paths)
    have = set(man.loc[man["status"].isin(["downloaded", "registered"]), "doc_id"])
    need = frame.assign(doc_id=[doc_id_for(f, y) for f, y in zip(frame["firm_id"], frame["fy"])])
    need = need[~need["doc_id"].isin(have)].drop_duplicates("doc_id")
    order = {s: i for i, s in enumerate(sources)}
    lst = listing[listing["source"].isin(sources)].copy()
    lst["prio"] = lst["source"].map(order)

    records, clients, n = [], {}, 0
    try:
        for _, row in need.iterrows():
            if limit and n >= limit:
                break
            opts = lst[(lst["firm_id"] == row["firm_id"]) & (lst["fy"].astype(int) == int(row["fy"]))]
            if opts.empty:
                continue
            dest = target_path(paths, row["firm_id"], row["fy"])
            for _, o in opts.sort_values("prio").iterrows():
                src = o["source"]
                try:
                    if src not in clients:
                        clients[src] = NseClient(paths) if src == "nse" else BseClient(cfg)
                    ok = clients[src].download(o["url"], dest)
                except Exception as exc:  # noqa: BLE001
                    log.warning("download failed %s %s: %s", row["doc_id"], o["url"], exc)
                    ok = False
                if ok:
                    records.append({"doc_id": row["doc_id"], "firm_id": row["firm_id"], "fy": row["fy"],
                                    "source": src, "url": o["url"], "local_path": _rel(dest),
                                    "pub_date": o.get("pub_date", ""), "sha256": _sha256(dest),
                                    "n_bytes": dest.stat().st_size, "status": "downloaded", "note": ""})
                    n += 1
                    log.info("downloaded %s from %s", row["doc_id"], src)
                    break
    finally:
        for c in clients.values():
            c.close()
    man = upsert_manifest(paths, records)
    still = set(need["doc_id"]) - set(r["doc_id"] for r in records)
    log.info("downloaded %d new PDFs; %d firm-years still missing (collect manually)", len(records), len(still))
    return man
