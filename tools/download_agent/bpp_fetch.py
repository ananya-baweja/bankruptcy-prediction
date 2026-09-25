#!/usr/bin/env python3
"""BPP download agent for the bankruptcy-prediction project.

Start it by double-clicking START_DOWNLOADER.bat in this folder (or run
`python bpp_fetch.py`). It watches the jobs\\ folder for job files that
Claude writes, downloads what each job lists into THIS folder, and logs every
request under logs\\. Leave the window open while downloads are needed; close
it (or press Ctrl+C) to stop. It resumes where it left off when restarted.

Safety:
  * it only contacts the public hosts listed in ALLOWED_HOSTS (IBBI, BSE, NSE,
    and the Notre Dame / Google Drive link of the Loughran-McDonald dictionary);
  * it only writes inside the folder this script is in;
  * it never opens or runs anything it downloads.

Standard library only - nothing to install.

1.1: several downloads at once (``workers`` per job) under one shared,
per-website speed limit; restarts itself when Claude updates this file
(START_DOWNLOADER.bat relaunches it on exit code 3).
1.4: a file counts as downloaded only when it is complete (Content-Length
matches; a PDF ends with %%EOF); otherwise it is downloaded again, up to 3 times.
1.5: "split" items cut a file that is too large to hand over in one piece into
numbered parts under transfer\ (bytes only - the file is never opened as a document).
1.6: a split can gzip the bytes first ("compress": "gzip"), which makes the
collected JSON-lines files about ten times smaller to hand over.
"""
import csv
import datetime as _dt
import hashlib
import http.cookiejar
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
import zlib
from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "1.6"
BASE = Path(__file__).resolve().parent
JOBS = BASE / "jobs"
DONE = JOBS / "done"
LOGS = BASE / "logs"
POLL_S = 20
TIMEOUT_S = 60
CHUNK = 1 << 20

ALLOWED_HOSTS = {
    "ibbi.gov.in", "www.ibbi.gov.in",
    "www.bseindia.com", "bseindia.com", "api.bseindia.com",
    "www.nseindia.com", "nseindia.com", "nsearchives.nseindia.com", "archives.nseindia.com",
    # Loughran-McDonald dictionary: sraf.nd.edu links to a Google Drive file (approved 2026-09-23)
    "sraf.nd.edu", "drive.google.com", "drive.usercontent.google.com",
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
REFERER = {
    "bseindia.com": "https://www.bseindia.com/",
    "nseindia.com": "https://www.nseindia.com/",
    "ibbi.gov.in": "https://ibbi.gov.in/",
}
RETRY_STATUS = {429, 500, 502, 503, 504}
RESTART_CODE = 3


def now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def say(msg):
    # ASCII only: the Windows console chokes on some characters.
    print(f"[{_dt.datetime.now():%H:%M:%S}] " + str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def site_of(host):
    for key in REFERER:
        if host == key or host.endswith("." + key):
            return key
    return None


def safe_path(rel):
    """Resolve a job-supplied relative path, refusing anything outside BASE."""
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("empty path")
    rel = rel.replace("\\", "/")
    if rel.startswith("/") or ":" in rel or any(part == ".." for part in rel.split("/")):
        raise ValueError(f"unsafe path: {rel!r}")
    # abspath, not resolve(): on Windows resolve() can return a differently
    # spelled path mid-write and refused a legitimate file (1.2, one PDF).
    # ".." is already refused above, so no symlink games are possible here.
    target = Path(os.path.abspath(BASE / rel))
    base_s = os.path.normcase(str(BASE)).rstrip("\\/") + os.sep
    if not os.path.normcase(str(target)).startswith(base_s):
        raise ValueError(f"path escapes the folder: {rel!r}")
    return target


def check_url(url):
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(f"only https is allowed: {url!r}")
    host = (parts.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"host not allowed: {host!r}")
    return host


class _Handled(Exception):
    """Internal: an item finished without a download (e.g. a split)."""


class Agent:
    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.next_slot = {}
        self.gate = threading.Lock()
        self.nse_lock = threading.Lock()
        self.nse_warm = 0.0
        self.started_mtime = Path(__file__).stat().st_mtime
        for d in (JOBS, DONE, LOGS):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ http
    def headers_for(self, host, extra=None):
        h = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
        site = site_of(host)
        if site:
            h["Referer"] = REFERER[site]
            if host.startswith("api.") or "/api/" in (extra or {}).get("_path", ""):
                h["Origin"] = REFERER[site].rstrip("/")
        for k, v in (extra or {}).items():
            if not k.startswith("_"):
                h[k] = v
        return h

    def warm_nse(self):
        with self.nse_lock:
            self._warm_nse()

    def _warm_nse(self):
        if time.time() - self.nse_warm < 600:
            return
        try:
            req = urllib.request.Request("https://www.nseindia.com/", headers=self.headers_for("www.nseindia.com"))
            with self.opener.open(req, timeout=TIMEOUT_S) as r:
                r.read(1 << 16)
            say("NSE session cookies refreshed")
        except Exception as e:  # noqa: BLE001
            say(f"NSE warm-up failed ({e}); continuing")
        self.nse_warm = time.time()

    def polite_wait(self, host, delay):
        """At most one request start per ``delay`` seconds per website, across all workers."""
        with self.gate:
            now_t = time.time()
            slot = max(now_t, self.next_slot.get(host, 0.0))
            self.next_slot[host] = slot + delay + random.uniform(0, delay * 0.3)
        wait = slot - time.time()
        if wait > 0:
            time.sleep(wait)

    def open(self, url, host, delay, extra_headers=None):
        """Returns an open response; raises urllib errors after retries."""
        if host == "www.nseindia.com" and "/api/" in url:
            self.warm_nse()
        attempt = 0
        while True:
            attempt += 1
            self.polite_wait(host, delay)
            hdrs = self.headers_for(host, dict(extra_headers or {}, _path=urlsplit(url).path))
            req = urllib.request.Request(url, headers=hdrs)
            try:
                return self.opener.open(req, timeout=TIMEOUT_S)
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt < 4:
                    back = 5 * 3 ** (attempt - 1)
                    say(f"  HTTP {e.code}, retrying in {back}s")
                    time.sleep(back)
                    continue
                if e.code in (401, 403) and host == "www.nseindia.com" and attempt < 3:
                    self.nse_warm = 0.0
                    self.warm_nse()
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < 4:
                    back = 5 * 3 ** (attempt - 1)
                    say(f"  network error ({e}), retrying in {back}s")
                    time.sleep(back)
                    continue
                raise

    # ------------------------------------------------------------------ jobs
    def pending_jobs(self):
        jobs = []
        for p in sorted(JOBS.glob("*.json")):
            if not (DONE / (p.stem + ".done")).exists():
                jobs.append(p)
        return jobs

    def heartbeat(self, state, job=None, progress=None):
        """Best effort: a status file that happens to be open elsewhere must never stop a job."""
        data = {"time": now(), "state": state, "job": job, "progress": progress,
                "version": VERSION, "python": sys.version.split()[0]}
        try:
            tmp = LOGS / "agent_heartbeat.json.tmp"
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, LOGS / "agent_heartbeat.json")
        except OSError:
            pass

    def run_job(self, job_path):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        name = job.get("name") or job_path.stem
        items = job.get("items", [])
        delay = float(job.get("delay_s", 1.0))
        workers = max(1, min(int(job.get("workers", 3)), 6))
        log_path = LOGS / f"{job_path.stem}.csv"
        new_log = not log_path.exists()
        logf = open(log_path, "a", newline="", encoding="utf-8")
        log = csv.writer(logf)
        if new_log:
            log.writerow(["time", "key", "url", "path", "status", "http", "bytes", "sha256", "note"])
        lock = threading.Lock()
        collected = {}   # collect-file -> set of keys already present
        counts = {"ok": 0, "skipped": 0, "failed": 0, "done": 0}
        state = {"t_last": 0.0}
        say(f"Job {name}: {len(items)} items, {workers} at a time")

        def keys_for(target):
            with lock:
                keys = collected.get(str(target))
                if keys is None:
                    keys = self.read_keys(target)
                    collected[str(target)] = keys
                return keys

        def one(item):
            url = item.get("url", "")
            key = str(item.get("key", url))
            kind = item.get("kind", "save")
            row, outcome = None, "failed"
            try:
                if kind == "split":
                    status, nbytes, sha, note = self.split(item)
                    row = [now(), key, "", item["path"], status, "", nbytes, sha, note]
                    outcome = "ok" if status == "ok" else "failed"
                    raise _Handled()
                host = check_url(url)
                if kind == "save":
                    target = safe_path(item["path"])
                    if target.exists() and target.stat().st_size > 0 and not item.get("overwrite"):
                        outcome = "skipped"
                    else:
                        status, http, nbytes, sha, note = self.save(url, host, delay, target, item)
                        row = [now(), key, url, item["path"], status, http, nbytes, sha, note]
                        outcome = "ok" if status == "ok" else "failed"
                elif kind == "collect":
                    target = safe_path(item["collect"])
                    keys = keys_for(target)
                    with lock:
                        seen = key in keys and not item.get("overwrite")
                    if seen:
                        outcome = "skipped"
                    else:
                        status, http, nbytes, sha, note, rec = self.collect(url, host, delay, key, item)
                        with lock:
                            with open(target, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            keys.add(key)
                        row = [now(), key, url, item["collect"], status, http, nbytes, sha, note]
                        outcome = "ok" if status == "ok" else "failed"
                else:
                    raise ValueError(f"unknown kind {kind!r}")
            except _Handled:
                pass
            except Exception as e:  # noqa: BLE001
                row = [now(), key, url, item.get("path") or item.get("collect"), "error", "", "", "",
                       f"{type(e).__name__}: {e}"[:300]]
                say(f"  failed: {url or key} ({type(e).__name__}: {e})")
            with lock:
                if row:
                    log.writerow(row)
                counts[outcome] += 1
                counts["done"] += 1
                if time.time() - state["t_last"] > 10 or counts["done"] == len(items):
                    logf.flush()
                    prog = {"done": counts["done"], "total": len(items), "ok": counts["ok"],
                            "skipped": counts["skipped"], "failed": counts["failed"]}
                    self.heartbeat("working", name, prog)
                    say(f"  {counts['done']}/{len(items)}  ok={counts['ok']} skipped={counts['skipped']} "
                        f"failed={counts['failed']}")
                    state["t_last"] = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))
        logf.close()
        summary = {"job": name, "finished": now(), "items": len(items),
                   **{k: counts[k] for k in ("ok", "skipped", "failed")}}
        (DONE / (job_path.stem + ".done")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        say(f"Job {name} finished: {summary}")

    @staticmethod
    def read_keys(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        keys = set()
        if target.exists():
            with open(target, encoding="utf-8") as f:
                for line in f:
                    try:
                        keys.add(str(json.loads(line)["key"]))
                    except Exception:  # noqa: BLE001
                        pass
        return keys

    def save(self, url, host, delay, target, item):
        """Download to ``<target>.part`` and move into place only when complete.

        A server can close the connection early and still answer HTTP 200 (one
        annual report arrived as 98 KB of a 3.9 MB file). A download counts as
        complete only when it matches Content-Length (when sent) and, for a PDF,
        ends with ``%%EOF``; otherwise it is downloaded again, up to three times.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        expect = item.get("expect")
        last = None
        for attempt in range(1, 4):
            if attempt > 1:
                time.sleep(5 * attempt)
            h = hashlib.sha256()
            nbytes = 0
            try:
                resp = self.open(url, host, delay, item.get("headers"))
            except urllib.error.HTTPError as e:
                part.unlink(missing_ok=True)
                return "http_error", e.code, 0, "", str(e.reason)[:200]
            with resp as r:
                code = r.status
                clen = r.headers.get("Content-Length")
                try:
                    with open(part, "wb") as f:
                        while True:
                            buf = r.read(CHUNK)
                            if not buf:
                                break
                            f.write(buf)
                            h.update(buf)
                            nbytes += len(buf)
                except (IncompleteRead, OSError) as e:
                    part.unlink(missing_ok=True)
                    last = ("truncated", code, nbytes, "", f"{type(e).__name__} after {nbytes} bytes")
                    say(f"  download interrupted ({type(e).__name__}), attempt {attempt}/3")
                    continue
            if nbytes == 0:
                part.unlink(missing_ok=True)
                return "empty", code, 0, "", ""
            if clen and clen.isdigit() and int(clen) != nbytes:
                part.unlink(missing_ok=True)
                last = ("truncated", code, nbytes, "", f"got {nbytes} of {clen} bytes")
                say(f"  incomplete download ({nbytes} of {clen} bytes), attempt {attempt}/3")
                continue
            if expect == "pdf":
                with open(part, "rb") as f:
                    head = f.read(1024)
                    f.seek(max(0, nbytes - 2048))
                    tail = f.read()
                if b"%PDF" not in head:
                    snippet = head[:120].decode("latin-1", "replace").replace("\n", " ")
                    part.unlink(missing_ok=True)
                    return "not_pdf", code, nbytes, "", snippet
                if b"%%EOF" not in tail:
                    part.unlink(missing_ok=True)
                    last = ("truncated_pdf", code, nbytes, "", f"no %%EOF after {nbytes} bytes")
                    say(f"  PDF cut short at {nbytes} bytes, attempt {attempt}/3")
                    continue
            os.replace(part, target)
            return "ok", code, nbytes, h.hexdigest(), (f"complete on attempt {attempt}" if attempt > 1 else "")
        return last

    @staticmethod
    def split(item):
        """Cut ``path`` into ``transfer/<name>.001, .002, ...`` of ``chunk_mb`` each.

        For handing a file that is too large for one transfer to the processing
        session, which joins the parts and checks the SHA-256 written next to them.
        With ``"compress": "gzip"`` the bytes are gzipped first (parts are then
        named ``<name>.gz.001`` ...); the SHA-256 in the manifest is always that of
        the original file. Reads and writes bytes only; the file itself is never
        opened as a document.
        """
        src = safe_path(item["path"])
        if not src.is_file():
            return "missing", 0, "", "no such file"
        chunk = max(1, min(int(item.get("chunk_mb", 60)), 200)) << 20
        gz = item.get("compress") == "gzip"
        out_dir = safe_path(item.get("out_dir", "transfer"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = (item.get("name") or src.parent.name + "_" + src.name) + (".gz" if gz else "")
        comp = zlib.compressobj(6, zlib.DEFLATED, 31) if gz else None   # wbits 31 = gzip format
        h = hashlib.sha256()
        parts = []
        pending = bytearray()
        size = 0

        def emit(buf):
            part = out_dir / f"{stem}.{len(parts) + 1:03d}"
            part.write_bytes(buf)
            parts.append({"part": part.name, "bytes": len(buf), "sha256": hashlib.sha256(buf).hexdigest()})

        with open(src, "rb") as f:
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                size += len(buf)
                h.update(buf)
                pending += comp.compress(buf) if gz else buf
                while len(pending) >= chunk:
                    emit(bytes(pending[:chunk]))
                    del pending[:chunk]
        if gz:
            pending += comp.flush()
        while pending:
            emit(bytes(pending[:chunk]))
            del pending[:chunk]
        manifest = {"source": item["path"], "bytes": size, "sha256": h.hexdigest(),
                    "compress": "gzip" if gz else None, "parts": parts}
        (out_dir / f"{stem}.parts.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        packed = sum(p["bytes"] for p in parts)
        say(f"  split {item['path']} into {len(parts)} parts" + (f" ({size} -> {packed} bytes gzipped)" if gz else ""))
        return "ok", size, h.hexdigest(), f"{len(parts)} parts" + (f", gzip {packed} bytes" if gz else "")

    def collect(self, url, host, delay, key, item):
        body, http, note, ctype = b"", None, "", ""
        try:
            with self.open(url, host, delay, item.get("headers")) as r:
                http = r.status
                ctype = r.headers.get("Content-Type", "")
                body = r.read()
        except urllib.error.HTTPError as e:
            http, note = e.code, str(e.reason)[:200]
        text = body.decode("utf-8", "replace")
        rec = {"key": key, "url": url, "http": http, "content_type": ctype, "fetched_at": now(),
               "meta": item.get("meta"), "body": text}
        status = "ok" if http == 200 and body else ("http_error" if http != 200 else "empty")
        return status, http, len(body), hashlib.sha256(body).hexdigest() if body else "", note, rec

    # ------------------------------------------------------------------ loop
    def updated(self):
        try:
            return Path(__file__).stat().st_mtime != self.started_mtime
        except OSError:
            return False

    def loop(self):
        say(f"BPP download agent {VERSION} - folder: {BASE}")
        say("Watching the jobs folder. Leave this window open; close it to stop.")
        idle_said = False
        while True:
            if self.updated():
                say("A new version of the agent was saved - restarting.")
                sys.exit(RESTART_CODE)
            jobs = self.pending_jobs()
            if not jobs:
                self.heartbeat("idle")
                if not idle_said:
                    say("No pending jobs - waiting for new ones...")
                    idle_said = True
                time.sleep(POLL_S)
                continue
            idle_said = False
            for jp in jobs[:1]:
                try:
                    self.run_job(jp)
                except Exception:  # noqa: BLE001
                    err = traceback.format_exc()
                    say(f"Job {jp.name} crashed:\n{err}")
                    (LOGS / f"{jp.stem}.crash.txt").write_text(err, encoding="utf-8")
                    (DONE / (jp.stem + ".done")).write_text(json.dumps({"job": jp.stem, "crashed": now()}),
                                                            encoding="utf-8")


if __name__ == "__main__":
    try:
        Agent().loop()
    except KeyboardInterrupt:
        say("Stopped. Run START_DOWNLOADER.bat again to resume.")
