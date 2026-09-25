"""Split an annual report's text into the sections we model.

Sections extracted
------------------
mdna                 Management Discussion and Analysis
directors_report     Directors' / Board's Report
auditor_report       Independent Auditor's Report (standalone preferred over consolidated)
  basis_for_modified_opinion   "Basis for Qualified / Adverse Opinion", "Basis for Disclaimer of Opinion"
  going_concern                "Material Uncertainty Related to Going Concern"
  emphasis_of_matter           "Emphasis of Matter"
caro_annexure        Annexure A to the Independent Auditor's Report (CARO: loan defaults, statutory dues)

How it works (rule-based, so it is explainable in a viva)
--------------------------------------------------------
1. **Heading hits.** Every short, heading-like line (or a short line joined with
   the next one, for headings that wrap) is matched against regex patterns for
   the sections above plus "boundary" sections that end them (Corporate
   Governance Report, Secretarial Audit Report, Balance Sheet, AGM Notice...).
2. **Filter false hits.**
   * Table-of-contents pages: an early page mentioning >= 3 different sections.
   * TOC-style lines: dotted leaders or a trailing page number.
   * Cross references: a heading followed by "forms part of", "annexed",
     "presented in a separate section"... (typical inside the Board's Report).
     Dropped only if the same section has another, real heading.
3. **Runs.** Hits are sorted by position; consecutive hits of the same section
   (e.g. running page headers) merge into one run that ends where a different
   section starts. A short run sandwiched between two runs of the same section
   is treated as a sub-heading and absorbed.
4. **Pick.** For MD&A / Directors' report: the longest run. For the auditor's
   report and CARO annexure: the first long run that is not about consolidated
   financial statements.
5. **Auditor sub-sections and opinion type** are found inside the auditor's report.
6. **Leakage counts.** Mentions of IBC/NCLT and CIRP-specific terms are counted
   so labelling can flag reports that may reveal the outcome.

Accuracy must be measured by hand on ~30 reports (``bpp qa-sample`` / ``bpp qa-score``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm import tqdm

from bpp.config import Paths

log = logging.getLogger(__name__)

TARGET_SECTIONS = ["mdna", "directors_report", "auditor_report", "caro_annexure"]
AUDITOR_SUBSECTIONS = ["basis_for_modified_opinion", "going_concern", "emphasis_of_matter"]

_AUD = r"auditor(?:'s|s'|s)?"
_ANNEX_LABEL = r"annexure\s*[-:.]?\s*['\"(]?\s*{x}\s*['\")]?\s*[-:]?\s*(?:to|of|referred\s+to\s+in)\s+(?:the\s+)?"

# (section, regex). Tried in order against the start of a heading candidate.
MAJOR_PATTERNS: list[tuple[str, str]] = [
    ("caro_annexure", _ANNEX_LABEL.format(x="(?:a|1|i)") + r"(?:independent\s+)?" + _AUD),
    ("caro_annexure", r"annexure\s*[-:.]?\s*['\"(]?\s*(?:a|1|i)\s*['\")]?\s*$"),   # "Annexure A" alone, joined later
    ("ifc_annexure", _ANNEX_LABEL.format(x="(?:b|2|ii)") + r"(?:independent\s+)?" + _AUD),
    # "Annexure to the Auditors' Report": no label - which one it is, the text decides
    ("caro_annexure", r"annexure\s*[-:]?\s*(?:to|of|referred\s+to\s+in)\s+(?:the\s+)?(?:independent\s+)?" + _AUD),
    ("auditor_report", r"independent\s+" + _AUD + r"\s+report"),
    ("auditor_report", _AUD + r"\s+report\s+to\s+the\s+members"),
    ("mdna", r"management(?:'s)?\s+discussion\s*(?:and|&)\s*analysis"),
    ("directors_report", r"(?:the\s+)?director(?:s'|'s|s)?\s+report"),
    ("directors_report", r"board(?:'s|s'|s)?\s+report"),
    ("directors_report", r"report\s+of\s+the\s+(?:board\s+of\s+)?directors"),
    ("corporate_governance", r"report\s+on\s+corporate\s+governance|corporate\s+governance\s+report"),
    ("secretarial_audit", r"(?:form\s+(?:no\.?\s*)?mr\s*-?\s*3\s*)?secretarial\s+audit\s+report"),
    ("business_responsibility", r"business\s+responsibility(?:\s+(?:and|&)\s+sustainability)?\s+report"),
    ("financial_statements", r"(?:standalone\s+|consolidated\s+)?balance\s+sheet\s+as\s+(?:at|on)"),
    ("financial_statements", r"(?:standalone\s+|consolidated\s+)?statement\s+of\s+profit\s+(?:and|&)\s+loss"),
    ("agm_notice", r"notice\s+of\s+(?:the\s+)?[\w\s-]{0,20}annual\s+general\s+meeting"),
]
_COMPILED = [(s, re.compile(p, re.I)) for s, p in MAJOR_PATTERNS]
_NO_PREFIX_STRIP = {"caro_annexure", "ifc_annexure"}

SUB_PATTERNS: list[tuple[str, str]] = [
    ("basis_for_modified_opinion",
     r"basis\s+for\s+(?:qualified|adverse)\s+opinion|basis\s+for\s+disclaimer\s+of\s+opinion"),
    ("going_concern", r"material\s+uncertainty\s+(?:related|relating)\s+to\s+(?:the\s+)?going\s+concern"),
    ("emphasis_of_matter", r"emphasis\s+of\s+matters?"),
    ("_boundary", r"key\s+audit\s+matters?|basis\s+for\s+opinion|(?:qualified\s+|adverse\s+|unmodified\s+)?opinion$"
                  r"|disclaimer\s+of\s+opinion$|information\s+other\s+than|other\s+information"
                  r"|management(?:'s)?\s+(?:and\s+board\s+of\s+directors(?:')?\s+)?responsibilit"
                  r"|responsibilit(?:y|ies)\s+of\s+(?:the\s+)?management|board\s+of\s+directors(?:'s?)?\s+responsibilit"
                  r"|" + _AUD + r"\s+responsibilit|report\s+on\s+other\s+legal|other\s+matters?$"
                  r"|report\s+on\s+the\s+audit"),
]
_SUB_COMPILED = [(s, re.compile(p, re.I)) for s, p in SUB_PATTERNS]

_NUMBERING = re.compile(r"^(?:\(?[ivxlc]{1,5}[.)]|\(?[a-z][.)]|\d{1,2}(?:\.\d{1,2})*[.)]?)\s+", re.I)
_ANNEX_PREFIX = re.compile(
    r"^(?:annexure|annex)\s*[-:]?\s*['\"(]?[a-z0-9]{1,5}['\")]?\s*[-:.]?\s*"
    r"(?:(?:to|of)\s+(?:the\s+)?(?:board(?:'s)?|directors(?:')?)\s+report\s*[-:.]?\s*)?", re.I)
_SENTENCE_WORDS = re.compile(
    r"\b(is|are|was|were|has|have|had|forms?|forming|annexed|attached|given|provided|prepared|includes?|"
    r"contains?|which|that|we|our|refer|should|shall|will)\b", re.I)
_TOC_LIKE = re.compile(r"\.{4,}|(?:\s|^)\d{1,3}\s*(?:-\s*\d{1,3})?\s*$")
_CROSSREF = re.compile(
    r"\b(forms?\s+(?:an?\s+)?(?:integral\s+)?part|forming\s+(?:an?\s+)?(?:integral\s+)?part|annexed|"
    r"attached\s+(?:herewith|as|to)|is\s+(?:given|provided|presented|set\s+out|included)\s+(?:in|as|at)|"
    r"separate\s+section|appended)\b", re.I)

_REFERS_TO_DOC = re.compile(r"\b(report|section|annexure|annual\s+report)\b", re.I)

#: what the CARO annexure talks about, and what the internal-financial-controls one does.
#: Auditors letter them either way round (CARO as "A" or "B", or "1", "I", or not at all),
#: so the text, not the letter, decides which is which.
_CARO_TEXT = re.compile(
    r"auditor'?s?'?\s*report\)\s*order|paragraphs?\s*3\s*(?:and|&)\s*4\s*of\s*the\s*(?:said\s+)?order"
    r"|physically\s+verified|title\s+deeds|statutory\s+dues|default\s+in\s+(?:the\s+)?repayment"
    r"|wilful\s+defaulter|tax\s+assessments|initial\s+public\s+offer|fraud\s+(?:by|on)\s+the\s+company"
    r"|nidhi\s+company|section\s*45-?\s*ia|managerial\s+remuneration|cost\s+records",
    re.I)
_IFC_TEXT = re.compile(
    r"internal\s+financial\s+controls?\s+(?:over|with\s+reference\s+to)\s+financial\s+(?:reporting|statements)"
    r"|section\s*143\s*\(\s*3\s*\)\s*\(\s*i\s*\)|clause\s*\(i\)\s*of\s*sub[-\s]*section\s*\(?3\)?\s*of\s*section\s*143",
    re.I)


def annexure_kind(following_text: str) -> str | None:
    """``caro_annexure``, ``ifc_annexure`` or None, from the text under an annexure heading."""
    head = following_text[:4000]
    nxt = re.search(r"\n\s*[\"'“‘\[]?\s*annexure\b", head[10:], re.I)     # stop at the next annexure
    if nxt:
        head = head[:10 + nxt.start()]
    caro = len(_CARO_TEXT.findall(head))
    ifc = len(_IFC_TEXT.findall(head[:1500]))
    if caro >= 2 and caro >= ifc:
        return "caro_annexure"
    if ifc >= 1:
        return "ifc_annexure"
    return "caro_annexure" if caro == 1 else None


def _is_crossref(rest_of_heading: str, first_sentence: str) -> bool:
    """A cross reference both points somewhere ("forms part of", "annexed"...) and, in the
    sentence after the heading, names a report/section/annexure, e.g.
    "The MD&A Report ... is presented in a separate section forming part of this Annual Report"."""
    both = rest_of_heading + " " + first_sentence
    return bool(_CROSSREF.search(both) and _REFERS_TO_DOC.search(first_sentence))


_IBC_GENERIC = re.compile(r"insolvency\s+and\s+bankruptcy\s+code|\bibc\b|national\s+company\s+law\s+tribunal|\bnclt\b",
                          re.I)
_CIRP_SPECIFIC = re.compile(
    r"corporate\s+insolvency\s+resolution\s+process|\bcirp\b|interim\s+resolution\s+professional|"
    r"\bresolution\s+professional\b|committee\s+of\s+creditors|moratorium\s+under\s+section\s+14|"
    r"section\s+(?:7|9|10)\s+of\s+the\s+(?:insolvency|ibc|code)|admitted\s+the\s+(?:petition|application)",
    re.I)


@dataclass
class Hit:
    section: str
    offset: int
    page: int
    heading: str
    toc_like: bool = False
    crossref: bool = False


@dataclass
class Run:
    section: str
    start: int
    end: int
    page: int
    heading: str
    hits: list[Hit] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


# ----------------------------------------------------------------------------- text assembly
def assemble(pages: list[dict[str, Any]]) -> tuple[str, list[int]]:
    """Join page texts; return the text and the start offset of each page."""
    parts, starts, pos = [], [], 0
    for p in pages:
        starts.append(pos)
        parts.append(p["text"])
        pos += len(p["text"]) + 2
    return "\n\n".join(parts), starts


def page_of(offset: int, starts: list[int]) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _classify_heading(candidate: str, compiled: list[tuple[str, re.Pattern]] = _COMPILED,
                      allow_prefix_strip: bool = True) -> tuple[str, str] | None:
    """Return (section, rest_of_line) if the candidate looks like a known heading."""
    cand = re.sub(r"\s+", " ", candidate).strip().strip("*#|:-").strip()
    cand = re.sub(r"^\[[^\]]{0,40}\]\s*", "", cand)                       # "[CIN: L25200...] Annexure 'A' ..."
    cand = cand.lstrip("\"'“”‘’ ")
    if not cand or len(cand) > 160:
        return None
    variants = [(cand, False)]
    stripped = _NUMBERING.sub("", cand)
    if stripped != cand:
        variants.append((stripped, False))
    if allow_prefix_strip:
        no_annex = _ANNEX_PREFIX.sub("", stripped)
        if no_annex != stripped:
            variants.append((no_annex, True))
    for text, prefix_stripped in variants:
        for section, rx in compiled:
            if prefix_stripped and section in _NO_PREFIX_STRIP:
                continue
            m = rx.match(text)
            if m:
                return section, text[m.end():]
    return None


def _is_heading_rest(rest: str) -> bool:
    rest = rest.strip()
    return len(rest) <= 45 and not _SENTENCE_WORDS.search(rest)


def find_heading_hits(text: str, starts: list[int], compiled=_COMPILED,
                      allow_prefix_strip: bool = True) -> list[Hit]:
    hits: list[Hit] = []
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        found = _classify_heading(line, compiled, allow_prefix_strip)
        used_two, next_line = False, ""
        if len(line) < 60:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_line = lines[j].strip()
                joined = _classify_heading(line + " " + next_line, compiled, allow_prefix_strip)
                bare = found is not None and found[0] == "caro_annexure" and found[1].strip() == ""
                if joined is not None and (found is None or (bare and joined[0] == "caro_annexure")):
                    found, used_two = joined, True
        if found is not None:
            section, rest = found
            bare_annex = section == "caro_annexure" and not used_two and rest.strip() == ""
            if section in ("caro_annexure", "ifc_annexure"):
                # the letter does not say which annexure it is; the text under it does. A
                # bare "Annexure A" is taken only when that text is an auditor's annexure
                # ("The Annexure referred to in paragraph 1 ..." follows it, not a heading).
                kind = annexure_kind(text[offsets[i]:offsets[i] + 4500])
                if kind is not None:
                    section, bare_annex = kind, False
                elif bare_annex or not re.search(_AUD, line + " " + next_line, re.I):
                    i += 1
                    continue
            annex_rest_ok = section in ("caro_annexure", "ifc_annexure") and len(rest.strip()) <= 120 \
                and not _SENTENCE_WORDS.search(rest)
            if (_is_heading_rest(rest) or annex_rest_ok) and not bare_annex:
                follow = text[offsets[i] + len(lines[i]):offsets[i] + len(lines[i]) + 450]
                follow = re.split(r"(?<=[a-z]{3})\.\s", follow, maxsplit=1)[0]   # first sentence only
                hits.append(Hit(section=section, offset=offsets[i], page=page_of(offsets[i], starts),
                                heading=f"{line} {next_line}" if used_two else line,
                                toc_like=bool(_TOC_LIKE.search(rest)),
                                crossref=_is_crossref(rest, follow)))
        i += 1
    return hits


def filter_hits(hits: list[Hit], cfg: dict[str, Any]) -> list[Hit]:
    ecfg = cfg["extraction"]
    by_page: dict[int, set[str]] = {}
    for h in hits:
        by_page.setdefault(h.page, set()).add(h.section)
    toc_pages = {p for p, s in by_page.items()
                 if p <= ecfg["toc_max_page"] and len(s) >= ecfg["toc_min_distinct_sections"]}
    kept = [h for h in hits if h.page not in toc_pages and not h.toc_like]
    # cross references: drop if the same section has a real heading elsewhere
    real = {h.section for h in kept if not h.crossref}
    return [h for h in kept if not (h.crossref and h.section in real)]


def build_runs(hits: list[Hit], doc_len: int, min_chars: int) -> list[Run]:
    hits = sorted(hits, key=lambda h: h.offset)

    def collapse(hs: list[Hit]) -> list[Run]:
        runs: list[Run] = []
        for h in hs:
            if runs and runs[-1].section == h.section:
                runs[-1].hits.append(h)
            else:
                runs.append(Run(h.section, h.offset, doc_len, h.page, h.heading, [h]))
        for a, b in zip(runs, runs[1:]):
            a.end = b.start
        if runs:
            runs[-1].end = doc_len
        return runs

    runs = collapse(hits)
    changed = True
    while changed:
        changed = False
        for k in range(1, len(runs) - 1):
            r = runs[k]
            if r.length < min_chars and runs[k - 1].section == runs[k + 1].section != r.section:
                drop = {id(h) for h in r.hits}
                hits = [h for h in hits if id(h) not in drop]
                runs = collapse(hits)
                changed = True
                break
    return runs


def _section_payload(text: str, run: Run, starts: list[int]) -> dict[str, Any]:
    body = text[run.start:run.end].strip()
    return {"heading": run.heading, "start_page": run.page,
            "end_page": page_of(max(run.end - 1, run.start), starts),
            "n_chars": len(body), "text": body}


def _is_consolidated(body: str) -> bool:
    head = body[:2500].lower()
    return "consolidated financial statements" in head and "standalone financial statements" not in head


def extract_auditor_subsections(aud_text: str) -> tuple[dict[str, Any], str]:
    """Find sub-sections and the opinion type inside the auditor's report text."""
    lines = aud_text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    hits: list[tuple[str, int, str]] = []
    headings_lower = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > 120:
            continue
        cand = _NUMBERING.sub("", re.sub(r"^[^\w(]+", "", s))          # "• Emphasis of Matter"
        cand = _unsplit_ligatures(cand)
        headings_lower.append(_heading_form(cand))
        for name, rx in _SUB_COMPILED:
            m = rx.match(cand)
            if m and _is_heading_rest(cand[m.end():]):
                hits.append((name, offsets[i], s))
                break
    subs: dict[str, Any] = {}
    for k, (name, off, heading) in enumerate(hits):
        if name == "_boundary":
            continue
        end = hits[k + 1][1] if k + 1 < len(hits) else len(aud_text)
        body = aud_text[off:end].strip()
        if name not in subs or len(body) > subs[name]["n_chars"]:
            subs[name] = {"heading": heading, "n_chars": len(body), "text": body}

    low = re.sub(r"\s+", " ", _unsplit_ligatures(aud_text.lower()))
    if any(re.fullmatch(r"disclaimer\s+of\s+opinion", h) for h in headings_lower) or \
            "we do not express an opinion" in low:
        opinion = "disclaimer"
    elif any(re.fullmatch(r"adverse\s+opinion", h) for h in headings_lower) or "basis for adverse opinion" in low:
        opinion = "adverse"
    elif any(re.fullmatch(r"qualified\s+opinion", h) for h in headings_lower) or "basis for qualified opinion" in low:
        opinion = "qualified"
    elif any(re.fullmatch(r"(?:unmodified\s+)?opinion", h) for h in headings_lower):
        opinion = "unmodified"
    # No opinion heading could be read (bullets, headings run into the line before,
    # OCR, the pre-2018 layout): the opinion paragraph's own wording decides.
    elif re.search(r"do(?:es)? not give a true and fair view", low):
        opinion = "adverse"
    elif re.search(r"except for the (?:possible )?effects? of the matters?", low):
        opinion = "qualified"
    elif re.search(r"give[s]? a true and fair view", low):
        opinion = "unmodified"
    else:
        opinion = "unknown"
    return subs, opinion


def _unsplit_ligatures(text: str) -> str:
    """PDF text splits the fi/fl ligatures: "Qualifi ed Opinion", "modifi ed"."""
    return re.sub(r"(?<=[a-z])(fi|fl) (?=(?:ed|es|er|cation|cations|nancial|rm|ve|x|ne)\b)", r"\1", text)


def _heading_form(line: str) -> str:
    """A heading line reduced to its words: bullets and colons gone, a doubled heading
    ("Opinion Opinion Opinion") once, and a heading run into the one before it
    ("...Standalone Financial Statements Opinion") cut to its last part."""
    h = _unsplit_ligatures(line.lower())
    h = re.sub(r"[^a-z\s]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    h = re.sub(r"\b(\w+)(?: \1\b)+", r"\1", h)                     # "opinion opinion opinion"
    h = re.sub(r"\b(\w+ \w+)(?: \1\b)+", r"\1", h)                 # "basis for opinion basis for opinion"
    m = re.search(r"(?:financial statements|report on the audit[a-z ]*?) ((?:qualified |adverse |unmodified )?opinion|"
                  r"disclaimer of opinion)$", h)
    return m.group(1) if m else h


def segment_document(pages: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    min_chars = cfg["extraction"]["min_section_chars"]
    text, starts = assemble(pages)
    hits = filter_hits(find_heading_hits(text, starts), cfg)
    runs = build_runs(hits, len(text), min_chars)
    warnings: list[str] = []
    sections: dict[str, Any] = {}

    for name in ["mdna", "directors_report"]:
        cands = [r for r in runs if r.section == name]
        if not cands:
            sections[name] = None
            warnings.append(f"{name}: not found")
            continue
        best = max(cands, key=lambda r: r.length)
        sections[name] = _section_payload(text, best, starts)
        sections[name]["n_candidate_runs"] = len(cands)
        if best.length < min_chars:
            warnings.append(f"{name}: short ({best.length} chars)")

    for name in ["auditor_report", "caro_annexure"]:
        cands = [r for r in runs if r.section == name]
        long_ = [r for r in cands if r.length >= min_chars] or cands
        if not long_:
            sections[name] = None
            warnings.append(f"{name}: not found")
            continue
        standalone = [r for r in long_ if not _is_consolidated(text[r.start:r.end])]
        chosen = standalone[0] if standalone else long_[0]
        sections[name] = _section_payload(text, chosen, starts)
        sections[name]["n_candidate_runs"] = len(cands)
        if name == "auditor_report":
            sections[name]["scope"] = "consolidated" if not standalone else "standalone"
            if not standalone:
                warnings.append("auditor_report: only a consolidated report was found")

    audit_opinion = "not_found"
    if sections.get("auditor_report"):
        subs, audit_opinion = extract_auditor_subsections(sections["auditor_report"]["text"])
        for s in AUDITOR_SUBSECTIONS:
            sections[s] = subs.get(s)
    else:
        for s in AUDITOR_SUBSECTIONS:
            sections[s] = None

    return {
        "sections": sections,
        "audit_opinion": audit_opinion,
        "leakage": {"ibc_generic_mentions": len(_IBC_GENERIC.findall(text)),
                    "cirp_specific_mentions": len(_CIRP_SPECIFIC.findall(text))},
        "runs": [{"section": r.section, "start_page": r.page, "n_chars": r.length, "heading": r.heading}
                 for r in runs],
        "warnings": warnings,
    }


#: "for the year ended 31st March, 2019", "year ended March 31, 2019", "period ended 31.03.2019"
_YEAR_ENDED = re.compile(
    r"(?:year|period)\s+ended\s+(?:on\s+)?(?:31\s*(?:st)?\s*(?:of\s+)?march[,\s]*|march\s*31\s*(?:st)?[,\s]*"
    r"|31[./-]0?3[./-])((?:19|20)\d\d)", re.I)


def report_year_check(pages: list[dict[str, Any]], filed_fy: int | None, min_mentions: int = 3) -> dict[str, Any]:
    """Whether a report's own text is about the year it is filed under.

    The exchange can list a report under the wrong year: on the full cohort one
    company's "FY2019" report was its 2017-18 report and another's "FY2020" and
    "FY2021" reports were its FY2019 and FY2020 reports. A report that never names its
    filed year in a "year ended 31 March" phrase while naming an earlier one at least
    ``min_mentions`` times is an earlier year's report; one that names a later year
    more often than its own is a later year's (and its real publication date is later
    than assumed, a leakage risk).
    """
    counts: dict[int, int] = {}
    for page in pages:
        for m in _YEAR_ENDED.finditer(page.get("text") or ""):
            year = int(m.group(1))
            counts[year] = counts.get(year, 0) + 1
    named = [y for y, n in counts.items() if n >= min_mentions]
    newest = max(named) if named else None
    own = counts.get(int(filed_fy), 0) if filed_fy else 0
    mismatch = ""
    if filed_fy and newest is not None:
        if newest < int(filed_fy) and own == 0:
            mismatch = "earlier_year_report"
        elif newest > int(filed_fy) and counts.get(newest, 0) > own:
            mismatch = "later_year_report"
    top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:4])
    return {"filed_fy": filed_fy, "filed_fy_mentions": own, "newest_named_fy": newest,
            "year_mentions": {str(k): v for k, v in top.items()}, "mismatch": mismatch}


def run_extract_sections(cfg: dict[str, Any], paths: Paths, doc_ids: list[str] | None = None,
                         force: bool = False) -> pd.DataFrame:
    files = sorted(paths.pages.glob("*.json"))
    if doc_ids:
        files = [f for f in files if f.stem in set(doc_ids)]
    n = checked = 0
    for f in tqdm(files, desc="extract sections"):
        out = paths.sections / f.name
        if out.exists() and not force:
            # sections already cut; add the year check to files written before it existed
            js = json.loads(out.read_text(encoding="utf-8"))
            if "year_check" not in js:
                pj = json.loads(f.read_text(encoding="utf-8"))
                js["year_check"] = report_year_check(pj["pages"], pj.get("fy"))
                out.write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
                checked += 1
            continue
        pj = json.loads(f.read_text(encoding="utf-8"))
        seg = segment_document(pj["pages"], cfg)
        res = {k: pj[k] for k in ["doc_id", "firm_id", "fy", "source_pdf", "n_pages", "n_ocr_pages",
                                  "n_needs_ocr_pages"] if k in pj}
        res.update(seg)
        res["year_check"] = report_year_check(pj["pages"], pj.get("fy"))
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    log.info("segmented %d documents (year check added to %d already segmented)", n, checked)
    return build_extraction_report(paths)


def build_extraction_report(paths: Paths) -> pd.DataFrame:
    from bpp.common import write_csv

    rows = []
    for f in sorted(paths.sections.glob("*.json")):
        js = json.loads(f.read_text(encoding="utf-8"))
        row = {k: js.get(k) for k in ["doc_id", "firm_id", "fy", "n_pages", "n_ocr_pages", "n_needs_ocr_pages",
                                      "audit_opinion"]}
        for name in TARGET_SECTIONS + AUDITOR_SUBSECTIONS:
            sec = js["sections"].get(name)
            row[f"{name}_chars"] = sec["n_chars"] if sec else 0
        if js["sections"].get("auditor_report"):
            row["auditor_scope"] = js["sections"]["auditor_report"].get("scope")
        row.update(js.get("leakage", {}))
        yc = js.get("year_check") or {}
        row["newest_year_named"] = yc.get("newest_named_fy")
        row["report_year_mismatch"] = yc.get("mismatch", "")
        row["warnings"] = "; ".join(js.get("warnings", []))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        write_csv(df, paths.extraction_report)
        found = {s: f"{(df[f'{s}_chars'] > 0).mean():.0%}" for s in TARGET_SECTIONS}
        log.info("section found-rate over %d docs: %s", len(df), found)
    return df
