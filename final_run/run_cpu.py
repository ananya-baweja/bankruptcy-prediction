#!/usr/bin/env python3
"""All CPU models on the frozen folds -> results_cpu/cpu_oof.csv (+ the mined lexicon)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "vendor")); sys.path.append(str(HERE.parent / "src"))
from bppfinal import cpu_models as cm  # noqa: E402

from bppfinal.load import table  # noqa: E402
OUT = HERE / "results_cpu"; OUT.mkdir(exist_ok=True)
df, R, L = table(sys.argv[1] if len(sys.argv) > 1 else None)
FS = cm.feature_sets(R, L)
STRICT = df.strict.to_numpy().astype(bool)
json.dump(FS, open(OUT / "feature_sets.json", "w"), indent=1)

t0 = time.time()
P = {}
P["altman"] = cm.altman_oof(df)
P["logit_ratios"] = cm.logit_oof(df, R)
P["hgb_ratios"] = cm.hgb_oof(df, R)
P["hgb_ratios_fm"] = cm.hgb_oof(df, R + ["financials_missing"])
P["hgb_both"] = cm.hgb_oof(df, R + L)
P["hgb_ratios_langtext"] = cm.hgb_oof(df, R + FS["lang_text"])
P["hgb_lang_fm"] = cm.hgb_oof(df, FS["lang_text"] + ["financials_missing"])
P["logit_both"] = cm.logit_oof(df, R + L)
P["hgb_lang"] = cm.hgb_oof(df, FS["lang_text"])
P["logit_lang"] = cm.logit_oof(df, FS["lang_text"])
P["hgb_mdna"] = cm.hgb_oof(df, FS["mdna"])
P["hgb_aud"] = cm.hgb_oof(df, FS["aud"])
P["hgb_flags"] = cm.hgb_oof(df, FS["flags"])
P["hgb_ratios_mdna"] = cm.hgb_oof(df, R + FS["mdna"])
P["hgb_ratios_aud"] = cm.hgb_oof(df, R + FS["aud"])
P["hgb_ratios_flags"] = cm.hgb_oof(df, R + FS["flags"])
P["hgb_both_nodrift"] = cm.hgb_oof(df, R + [c for c in L if c not in FS["drift"]])
P["hgb_both_noperp"] = cm.hgb_oof(df, R + [c for c in L if c != "perplexity_fold"])
P["hgb_ratios_strict"] = cm.hgb_oof(df, R, rows=STRICT)
P["hgb_both_strict"] = cm.hgb_oof(df, R + L, rows=STRICT)
P["perplexity"] = df.perplexity_fold.to_numpy(float)
print(f"tabular models {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
P["tfidf_mdna"] = cm.tfidf_oof(df, "mdna")
P["tfidf_aud"] = cm.tfidf_oof(df, "auditor_report")
print(f"tf-idf {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
lex, lists = cm.lexicon_oof(df, "mdna")
lexa, lists_a = cm.lexicon_oof(df, "auditor_report")
P["lexicon_mdna"] = lex; P["lexicon_aud"] = lexa
df2 = df.assign(lexicon_mdna=lex, lexicon_aud=lexa)
P["hgb_both_lex"] = cm.hgb_oof(df2, R + L + ["lexicon_mdna", "lexicon_aud"])
P["hgb_lang_lex"] = cm.hgb_oof(df2, FS["lang_text"] + ["lexicon_mdna", "lexicon_aud"])
json.dump({str(k): {"distress": v[0], "healthy": v[1]} for k, v in lists.items()},
          open(OUT / "lexicon_fold_lists.json", "w"), indent=1)
full = cm.lexicon_full(df, "mdna")
full.to_csv(OUT / "india_distress_lexicon_mdna.csv", index=False)
full_a = cm.lexicon_full(df, "auditor_report")
full_a.to_csv(OUT / "india_distress_lexicon_auditor.csv", index=False)
print(f"lexicon {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame({"doc_id": df.doc_id, "label": df.label, "fold": df.fold, "strict": df.strict,
                    "horizon": df.horizon, **P})
out.to_csv(OUT / "cpu_oof.csv", index=False)
print("saved", OUT / "cpu_oof.csv", out.shape)
