"""Link IBBI debtors to their BSE/NSE listing: CIN first, names second, the report last.

Approved by the team on 2026-09-23 as a change to Phase 1 (decisions log):

1. **Who is listed** is read off the debtor's CIN (``L`` = listed at some point)
   instead of being inferred from whether a fuzzy name match exists.
2. **Which listing** is found by name against the BSE (active, suspended,
   delisted) and NSE lists, with the same scorer the Phase 1 code uses
   (rapidfuzz-compatible ``token_sort_ratio`` on normalised names).
3. **Confirmation** comes from the company's own annual report: once a report
   of the matched listing is downloaded, the CIN printed in it must equal the
   IBBI CIN (``confirm_with_report_cin``). Only mismatches and no-CIN cases need a
   person.

rapidfuzz is not a dependency of this module: the Indel ratio below is exact
(bit-parallel LCS), and candidates are short-listed by shared character trigrams
first, which never drops a candidate that could score above ~60.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from bpp.common import normalize_company_name

CIN_IN_TEXT = re.compile(r"\b([LU])\s?(\d{5})\s?([A-Z]{2})\s?(\d{4})\s?([A-Z]{3})\s?(\d{6})\b")


# ----------------------------------------------------------------- scoring
def lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence (Hyyro's bit-parallel algorithm)."""
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return 0
    peq: dict[str, int] = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    mask = (1 << m) - 1
    v = mask
    for ch in b:
        u = v & peq.get(ch, 0)
        v = ((v + u) | (v - u)) & mask
    return m - bin(v).count("1")


def ratio(a: str, b: str) -> float:
    """rapidfuzz ``fuzz.ratio``: normalised Indel similarity, 0-100."""
    if not a and not b:
        return 100.0
    return 200.0 * lcs_len(a, b) / (len(a) + len(b))


def token_sort_ratio(a: str, b: str) -> float:
    return ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _trigrams(s: str) -> set[str]:
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


class NameIndex:
    """Trigram index over the listed universe for fast candidate short-lists."""

    def __init__(self, names: Iterable[str]):
        self.names = list(names)
        self.sorted = [" ".join(sorted(n.split())) for n in self.names]
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, n in enumerate(self.sorted):
            for g in _trigrams(n):
                self.index[g].append(i)
        n_docs = max(len(self.names), 1)
        self.idf = {g: math.log(n_docs / len(ix)) for g, ix in self.index.items()}

    def search(self, query: str, limit: int = 3, shortlist: int = 60) -> list[tuple[float, int]]:
        q = " ".join(sorted(query.split()))
        if not q:
            return []
        scores: Counter = Counter()
        for g in _trigrams(q):
            w = self.idf.get(g)
            if w is None:
                continue
            for i in self.index[g]:
                scores[i] += w
        cands = [i for i, _ in scores.most_common(shortlist)]
        scored = sorted(((ratio(q, self.sorted[i]), i) for i in cands), key=lambda t: -t[0])
        return scored[:limit]


# ----------------------------------------------------------------- matching
def _bracketed_score(a: str, b: str) -> float:
    """Similarity of two company names with the words inside brackets kept."""
    keep = lambda n: normalize_company_name(re.sub(r"[()\[\]]", " ", str(n or "")))  # noqa: E731
    return token_sort_ratio(keep(a), keep(b))


def match_debtors(debtors: pd.DataFrame, universe: pd.DataFrame, auto_accept: float = 98,
                  review: float = 80, top_k: int = 3) -> pd.DataFrame:
    """Best listing for each debtor, in the columns of ``ibbi_listed_matches.csv``.

    ``match_status``: ``auto_accepted`` (score >= auto_accept), ``needs_review``
    (review <= score < auto_accept), ``no_match``. For a debtor with an ``L`` CIN
    the review band is where the report-CIN check does the reviewing.
    """
    uni = universe.dropna(subset=["name_norm"]).reset_index(drop=True)
    index = NameIndex(uni["name_norm"])
    out = []
    for _, d in debtors.iterrows():
        queries = [d["name_norm"]]
        for alias in str(d.get("former_names") or "").split(" | "):
            if alias.strip():
                queries.append(normalize_company_name(alias))
        for other in str(d.get("all_names") or "").split(" | "):
            n = normalize_company_name(other)
            if n and n not in queries:
                queries.append(n)
        best: dict[int, float] = {}
        for q in queries:
            for score, i in index.search(q, limit=top_k + 2):
                best[i] = max(best.get(i, 0.0), score)
        # Names that differ only in brackets tie after normalisation: IBBI's insolvent
        # "Asian Hotels (West) Limited" scored 100 against Asian Hotels (East), (North)
        # and (West), and the first of them was accepted (a wrong firm in the full
        # cohort). The bracketed words break such a tie; other orderings are unchanged.
        raw_names = [n for n in [d.get("corporate_debtor"), *str(d.get("all_names") or "").split(" | ")] if n]
        bracketed = {i: max((_bracketed_score(n, uni.iloc[i]["company_name"]) for n in raw_names), default=0.0)
                     for i in best}
        ranked = sorted(best.items(), key=lambda t: (-t[1], -bracketed[t[0]]))[:top_k]
        rec = {k: d.get(k) for k in ("corporate_debtor", "cin", "cirp_announcement_date",
                                     "n_announcements", "applicant", "listing_evidence",
                                     "cin_nic", "cin_company_type", "self_filed", "all_names")}
        rec["pa_pdf_url"] = None
        if ranked:
            i, score = ranked[0]
            u = uni.iloc[i]
            rec.update({
                "match_status": ("auto_accepted" if score >= auto_accept else
                                 "needs_review" if score >= review else "no_match"),
                "score": round(score, 1), "firm_id": u["firm_id"], "matched_name": u["company_name"],
                "bse_code": u.get("bse_code"), "nse_symbol": u.get("nse_symbol"), "isin": u.get("isin"),
                "listing_status": u.get("status"), "industry": u.get("industry"),
                "other_candidates": " | ".join(f"{uni.iloc[j]['company_name']} ({s:.0f})"
                                               for j, s in ranked[1:]),
            })
        else:
            rec.update({"match_status": "no_match", "score": 0.0})
        rec.update({"accept": "Y" if rec["match_status"] == "auto_accepted" else "",
                    "petition_date": "", "admission_date_verified": "", "business_group": "",
                    "report_cin": "", "report_cin_check": "", "notes": ""})
        out.append(rec)
    df = pd.DataFrame(out)
    order = {"needs_review": 0, "auto_accepted": 1, "no_match": 2}
    return df.sort_values(["match_status", "score"],
                          key=lambda s: s.map(order) if s.name == "match_status" else -s).reset_index(drop=True)


# ----------------------------------------------------------------- report check
def cins_in_text(text: str) -> list[str]:
    """Every CIN printed in a text, in order of appearance, de-duplicated."""
    seen = []
    for m in CIN_IN_TEXT.finditer(text.upper()):
        cin = "".join(m.groups())
        if cin not in seen:
            seen.append(cin)
    return seen


def confirm_with_report_cin(expected_cin: str | None, report_text: str) -> tuple[str, str]:
    """(status, cin found). status: confirmed | mismatch | cin_not_found | no_expected_cin.

    The first pages of an annual report print the company's own CIN, but also
    those of subsidiaries and the registrar; a match anywhere in the first pages
    confirms, and the first CIN found is reported otherwise.
    """
    found = cins_in_text(report_text)
    if not expected_cin:
        return "no_expected_cin", found[0] if found else ""
    if not found:
        return "cin_not_found", ""
    if expected_cin in found:
        return "confirmed", expected_cin
    # same company whose listing status changed keeps its registration number
    same_reg = [c for c in found if c[-6:] == expected_cin[-6:] and c[6:12] == expected_cin[6:12]]
    if same_reg:
        return "confirmed_reg_no", same_reg[0]
    return "mismatch", found[0]
