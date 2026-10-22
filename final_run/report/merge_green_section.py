#!/usr/bin/env python3
"""Inserts the Green AI study as Section 11 into the final report of 3 October 2026 and renumbers what
follows (Conclusion 12, Reproducibility 13, References 14; Table 22 -> 25), keeping Sections 1-10
exactly as built by build_report.js. Used because Node's docx package could not be installed when
the section was added; the standalone section builder is build_green_section.js.

    python merge_green_section.py REPORT_3OCT.docx GREEN_SECTION.docx RESULTS_GREEN_DIR OUT.docx

RESULTS_GREEN_DIR must hold green_results.csv, green_on_corrected_data.csv and pareto.png.
Needs python-docx and Pillow.
"""
import copy, re, sys, pandas as pd, docx
from docx.oxml.ns import qn
from PIL import Image
MAIN, GSEC, RG, OUT = sys.argv[1:5]
PARETO = f"{RG}/pareto.png"
W=qn
d=docx.Document(MAIN); g=docx.Document(GSEC)
B=list(d.element.body.iterchildren()); GB=list(g.element.body.iterchildren())
txt=lambda el:''.join(x.text or '' for x in el.iter(W('w:t')))
def find(pred):
    for el in d.element.body.iterchildren():
        if pred(txt(el)): return el
    raise KeyError
# ---------------- numbers
C=pd.read_csv(f"{RG}/green_on_corrected_data.csv",index_col=0)
E=pd.read_csv(f"{RG}/green_results.csv").set_index('config')
gwh=lambda c:1000*E.loc[c,'gpu_kwh']; gco=lambda c:1000*E.loc[c,'gpu_kg_co2e']
f3=lambda x:f"{x:.3f}"; sg=lambda x:("+" if x>=0 else "−")+f"{abs(x):.3f}"
sav=lambda c:100*(1-E.loc[c,'gpu_kwh']/E.loc['blend_finbert_fp16_k7','gpu_kwh'])
# ---------------- element builders
def run(text,rpr=None,bold=False,italic=False):
    r=d.element.makeelement(W('w:r'),{})
    rp=copy.deepcopy(rpr) if rpr is not None else d.element.makeelement(W('w:rPr'),{})
    if bold and rp.find(W('w:b')) is None:
        rp.insert(0,d.element.makeelement(W('w:bCs'),{})); rp.insert(0,d.element.makeelement(W('w:b'),{}))
    if italic and rp.find(W('w:i')) is None:
        rp.insert(0,d.element.makeelement(W('w:iCs'),{})); rp.insert(0,d.element.makeelement(W('w:i'),{}))
    if len(rp): r.append(rp)
    t=d.element.makeelement(W('w:t'),{}); t.text=text; t.set('{http://www.w3.org/XML/1998/namespace}space','preserve'); r.append(t); return r
def para(template,markup,rpr=None):
    p=d.element.makeelement(W('w:p'),{}); pp=template.find(W('w:pPr'))
    if pp is not None: p.append(copy.deepcopy(pp))
    for k,seg in enumerate(re.split(r'(\*\*.+?\*\*|\*.+?\*)',markup)):
        if not seg: continue
        if seg.startswith('**'): p.append(run(seg[2:-2],rpr,bold=True))
        elif seg.startswith('*'): p.append(run(seg[1:-1],rpr,italic=True))
        else: p.append(run(seg,rpr))
    return p
T_BODY=find(lambda t:t.startswith('Everything in this report can be regenerated'))
T_BUL=find(lambda t:t.startswith('Sample size is the binding'))
T_SUMBUL=find(lambda t:t.startswith("The auditor's report is the stronger text"))
T_CAP=find(lambda t:t.startswith('Table 22. Files'))
T_FIG=[el for el in d.element.body.iterchildren() if el.find('.//'+W('w:drawing')) is not None][-1]
T_FIGCAP=find(lambda t:t.startswith('Figure 10.'))
H1=find(lambda t:t=='11. Conclusion')
cap_r=[r.find(W('w:rPr')) for r in T_CAP.iter(W('w:r'))]
figcap_rpr=next(T_FIGCAP.iter(W('w:r'))).find(W('w:rPr'))
def caption(label,text):
    p=d.element.makeelement(W('w:p'),{}); p.append(copy.deepcopy(T_CAP.find(W('w:pPr'))))
    p.append(run(label+" ",cap_r[0])); p.append(run(text,cap_r[1])); return p
def heading(text):
    h=copy.deepcopy(H1)
    for r in list(h.iter(W('w:r')))[1:]: r.getparent().remove(r)
    next(h.iter(W('w:t'))).text=text; return h
def set_cell(cell,text):
    ps=cell.findall(W('w:p'))
    for p in ps[1:]: cell.remove(p)
    rs=ps[0].findall(W('w:r'))
    for r in rs[1:]: ps[0].remove(r)
    ts=list(rs[0].iter(W('w:t')))
    for t in ts[1:]: t.getparent().remove(t)
    ts[0].text=text
def table_from(template,rows,bold_rows=()):
    t=copy.deepcopy(template); trs=t.findall(W('w:tr'))
    plain=trs[1]; boldrow=next((r for r in trs[1:] if r.find('.//'+W('w:b')) is not None),plain)
    for r in trs[1:]: t.remove(r)
    for i,vals in enumerate(rows):
        r=copy.deepcopy(boldrow if i in bold_rows else plain)
        for c,v in zip(r.findall(W('w:tc')),vals): set_cell(c,v)
        t.append(r)
    return t
# ---------------- Section 11 content
c=C
gb,cur=c.loc['GB ratios+language (chosen)'],c.loc['Blend 7 seed(s)']
R22=[]
def r22(name,key,cfg,ref=False):
    x=c.loc[key]; R22.append([name,f"{f3(x.pr)} ± {f3(x.pr_sd)}",f3(x.pr_strict),f3(x.roc),f"{gwh(cfg):.1f}",f"{gco(cfg):.2f}",
        "reference" if ref else f"{sg(x.lo)} to {sg(x.hi)}"])
r22("Full system: blend, FinBERT, 7 seeds (current)",'Blend 7 seed(s)','blend_finbert_fp16_k7',True)
r22("Blend, FinBERT, 5 seeds",'Blend 5 seed(s)','blend_finbert_fp16_k5')
r22("Blend, FinBERT, 3 seeds",'Blend 3 seed(s)','blend_finbert_fp16_k3')
r22("Blend, FinBERT, 1 seed",'Blend 1 seed(s)','blend_finbert_fp16_k1')
r22("Network only, FinBERT, 7 seeds",'Network only 7 seeds','dl_finbert_fp16_k7')
r22("GB, ratios + language, no network (chosen)",'GB ratios+language (chosen)','gb_both')
r22("GB, ratios only",'GB ratios only','gb_ratios')
# chosen-vs-current: use the report's own paired test (Table 11: blend - hgb_both) for exact consistency
REP=dict(diff=-0.012,lo=-0.031,hi=+0.009,wil=0.81,gb_wins=3)
R22[5][6]=f"{sg(REP['lo'])} to {sg(REP['hi'])}"
k3,k5,k1=c.loc['Blend 3 seed(s)'],c.loc['Blend 5 seed(s)'],c.loc['Blend 1 seed(s)']
cw,gw=gwh('blend_finbert_fp16_k7'),gwh('gb_both')
emb=1000*E.loc['blend_finbert_fp16_k7','gpu_kwh_embed']; dl=1000*E.loc['blend_finbert_fp16_k7','gpu_kwh_dl']
S=[]
S.append(heading("11. Green AI: accuracy versus energy cost"))
S.append(para(T_BODY,"After the main results were fixed, the project asked how much of its energy buys accuracy. The pipeline was re-run with every stage's energy measured by codecarbon (offline mode, India grid): building the table, embedding the MD&A, each network run, the gradient boosting and per-document scoring. **Energy is reported as GPU energy**, read from the NVIDIA driver (NVML) and confirmed by integrating nvidia-smi power once per second (within 3.4% on every GPU-active stage longer than 10 s), and **CO₂e is computed from GPU energy alone** with codecarbon's India grid factor (0.713 kg per kWh). The whole-machine figure, which adds the processor's rated power times its utilisation, is not used as a result: unrelated background processes kept about 11 of the GPU computer's 20 hardware threads busy throughout the runs, which inflates every processor figure by an unknown amount. GPU energy includes the card's idle draw of 6 to 7 W while the processor works. Whole-machine totals are given in Table 24 as upper bounds only."))
S.append(para(T_BODY,f"The study has two parts. **On the data of Section 7**, the configurations that differ only in how many networks are trained, or in whether a network is used at all, were scored from the saved out-of-fold predictions of the final run, so their accuracy is exactly that of Tables 10 and 11; their GPU energy is that of the measured runs, which perform the same computation (Table 22). The current system draws **{cw:.1f} Wh of GPU energy** to produce its five-fold predictions: {emb:.1f} Wh to embed every MD&A with FinBERT and {dl:.1f} Wh to train the seven networks. **The full study of {len(E)} configurations** (smaller encoders: MiniLM-L6, DistilRoBERTa and a finance-tuned DistilRoBERTa; fp32, fp16 and bf16 FinBERT; the first 32, 64 or 96 MD&A sentences, the network reading 96; 1 to 7 seeds; INT8 FinBERT on the processor; no network at all) needed new networks to be trained, and was run on the dataset before the CARO correction of Section 8 (Table 23 and Figure 11). Its accuracies are about 0.015 lower than Section 7's and are used only to compare configurations with each other; every configuration in it saw the same data."))
S.append(caption("Table 22.","On the data of Section 7: accuracy from the saved out-of-fold predictions of the final run (mean ± sd over the 5 folds; the 609 column scores the same predictions on the strict rows), GPU energy (NVML) to produce the five-fold predictions, CO₂e from GPU energy only. Δ PR-AUC: 95% pair-bootstrap interval of the difference from the full system."))
S.append(table_from(GB[4],R22,bold_rows=(5,)))
S.append(para(T_BODY,f"**The chosen configuration.** The selection rule was fixed before any variant was run: the lowest-energy configuration whose PR-AUC is within 0.010 of the current system, or whose pair-bootstrap interval against it includes zero (energies within 5% of the lowest count as a tie, broken by PR-AUC). On both versions of the data it selects **gradient boosting on the ratios and language features, with no neural network**. On the data of Section 7 it scores PR-AUC {f3(gb.pr)} against {f3(cur.pr)} for the full system ({f3(gb.pr_strict)} against {f3(cur.pr_strict)} on the strict 609 rows): a difference of {sg(REP['diff'])} whose pair-bootstrap interval ({sg(REP['lo'])} to {sg(REP['hi'])}) includes zero, with a per-fold Wilcoxon p of {REP['wil']:.2f}; gradient boosting is ahead in {REP['gb_wins']} of the 5 folds. Its GPU energy is {gw:.1f} Wh instead of {cw:.1f} Wh, all of it the idle card while the processor builds the table: **{sav('gb_both'):.1f}% less GPU energy and CO₂e, about {cw/gw:.0f} times less**. Per annual report, the current scorer spends 213 ms and 10.3 J of GPU energy on FinBERT and the seven networks; the chosen model needs no GPU and scores a report in about 2 ms on the processor. On these data the choice does not depend on the tie clause, because gradient boosting on ratios alone, the only cheaper candidate, does not qualify (interval {sg(c.loc['GB ratios only'].lo)} to {sg(c.loc['GB ratios only'].hi)}). The price is the small gain the network adds on top of the language features, which Section 7 already found to be statistically unclear."))
S.append(para(T_BODY,f"**If the network is kept**, the number of seeds is the cheapest lever, because training the seven networks takes {100*dl/cw:.0f}% of the GPU energy. Three seeds instead of seven change PR-AUC by {sg(k3['pr']-cur.pr)} (interval {sg(k3.lo)} to {sg(k3.hi)}) for {sav('blend_finbert_fp16_k3'):.0f}% less GPU energy; five seeds by {sg(k5['pr']-cur.pr)} for {sav('blend_finbert_fp16_k5'):.0f}% less; one seed by {sg(k1['pr']-cur.pr)} for {sav('blend_finbert_fp16_k1'):.0f}% less. In the full study a three-seed blend with DistilRoBERTa matched the current system ({sg(E.loc['blend_distilroberta_fp16_k3','pr_all']-E.loc['blend_finbert_fp16_k7','pr_all'])}) for {sav('blend_distilroberta_fp16_k3'):.0f}% less, and FinBERT on the first 32 MD&A sentences with one seed lost {abs(E.loc['blend_finbert_fp16_n32_k1','pr_all']-E.loc['blend_finbert_fp16_k7','pr_all']):.3f} for {sav('blend_finbert_fp16_n32_k1'):.0f}% less."))
# Table 23 + figure from the green section, captions rewritten
t23=copy.deepcopy(GB[4])
S.append(caption("Table 23.",f"The full study, on the data before the CARO correction (Section 8): accuracy and GPU energy of selected configurations (mean ± sd over the 5 folds; GPU energy, NVML; CO₂e from GPU energy only). Δ PR-AUC: 95% pair-bootstrap interval against the current blend of that study (PR-AUC {E.loc['blend_finbert_fp16_k7','pr_all']:.3f}). All {len(E)} configurations are in results/green/green_results.csv."))
S.append(t23)
# figure
rid,_=d.part.get_or_add_image(PARETO)
fig=copy.deepcopy(T_FIG); blip=next(fig.iter(qn('a:blip'))); blip.set(qn('r:embed'),rid)
w,h=Image.open(PARETO).size; ext=next(fig.iter(qn('wp:extent'))); cx=int(ext.get('cx')); cy=int(cx*h/w)
ext.set('cy',str(cy))
for e in fig.iter(qn('a:ext')): 
    if e.get('cx'): e.set('cx',str(cx)); e.set('cy',str(cy))
dp=next(fig.iter(qn('wp:docPr'))); dp.set('id','911'); dp.set('name','Figure 11')
S.append(fig)
S.append(para(T_FIGCAP,f"Figure 11. PR-AUC against GPU energy (log scale) for all {len(E)} configurations of the full study (data before the CARO correction). The dashed line is the Pareto front; numbered points are listed beside the plot.",figcap_rpr))
n_unread="24,522 of 77,523"
S.append(para(T_BODY,f"**What did not help much.** Smaller encoders keep the accuracy (MiniLM-L6, five times smaller: {E.loc['blend_minilm_fp16_k7','pr_all']:.3f} against {E.loc['blend_finbert_fp16_k7','pr_all']:.3f} in the full study), and fp16 embeddings match fp32 (cosine similarity above 0.9999) for 4.4 times less GPU energy in the embedding stage, but the encoder is the smaller share of the bill: the network runs are. The network reads only the first 96 sentence slots, so about a third of the sentences the pipeline embeds ({n_unread}) are never read; capping the embedding at 96 sentences saves energy without changing what the network sees. Caching the embeddings, which the pipeline already did, saves 8.7 Wh of GPU energy over re-embedding for every seed and 58.4 Wh over the original 45-run experiment grid. Notebook 1 now applies these choices: it embeds only the 96 sentences per section that the network reads, in fp16, with a cache, and records the energy of each heavy step (energy_log.csv); Part 2 and the results of Section 7 are unchanged."))
S.append(para(T_BODY,f"**INT8 on the processor: inconclusive.** Dynamic INT8 quantisation of FinBERT changed the sentence vectors substantially (cosine 0.74 to fp32), yet the blend trained on them still scored {E.loc['blend_finbert_int8_k7','pr_all']:.3f} in the full study; that accuracy result stands. Its cost does not: the INT8 embedding and the per-document processor benchmarks ran while the background processes held about 11 of the 20 hardware threads, so neither their speed nor their energy can be compared with the GPU figures. Those measurements are listed in Table 24 as upper bounds and are not used in any conclusion."))
S.append(para(T_BODY,"**Limitations of these numbers.**"))
for btxt in ["GPU energy is a driver reading; CO₂e applies codecarbon's India grid factor to it. It leaves out the processor and memory, so it understates the true energy of every configuration, most of all the processor-only ones; it is used because the processor figure cannot be separated from the background load.",
             "For the chosen configuration, the GPU figure is the card's idle draw while the processor works, so it is best read as \"no GPU work\" rather than as a precise number; a computer without a GPU would draw none.",
             "The India grid factor is an assumption; another year's or region's factor scales every CO₂ figure, not the percentages.",
             "All runs are on one computer (Intel i9-7900X, RTX 5060 Ti), once per configuration; early stopping makes a single network run's energy vary by up to 1.8 times between seeds. PDF reading and feature extraction, shared by every configuration, were not measured, so the true saving per report is smaller.",
             "The encoder, precision and sentence-cap comparisons of the full study use the data before the CARO correction. Every configuration there saw the same data, so the comparisons stand, but their absolute accuracies should not be quoted as the project's results; those are in Table 22 and Section 7."]:
    S.append(para(T_BUL,btxt))
t24=copy.deepcopy(GB[16])
for tr in t24.findall(W('w:tr')):
    tcs=tr.findall(W('w:tc'))
    if txt(tcs[0]).startswith('Blend, DistilRoBERTa'):   # no GPU scorer was benchmarked for it
        set_cell(tcs[5],'n/a'); set_cell(tcs[6],'n/a')
S.append(caption("Table 24.","Appendix: whole-machine energy, upper bounds only. codecarbon totals including the processor (rated power × utilisation) and memory, and per-report energy from the same benchmarks, all measured while unrelated processes held about 11 of the 20 hardware threads. They overstate every configuration, most of all those that run longest on the processor; no conclusion rests on them."))
S.append(t24)
for el in S: H1.addprevious(el)
# ---------------- renumbering and cross-references after the new section
def retext(el,pairs):
    for t in el.iter(W('w:t')):
        if t.text:
            s=t.text
            for a,b_ in pairs: s=s.replace(a,b_)
            t.text=s
body=list(d.element.body.iterchildren()); start=body.index(H1)
for el in body[start:]:
    retext(el,[("11. Conclusion","12. Conclusion"),("12. Reproducibility","13. Reproducibility"),("13. References","14. References"),("Table 22. ","Table 25. ")])
for el in body[:body.index(S[0])]:
    retext(el,[("Section 13","Section 14"),("Section 12","Section 13")])
# info table: report date
info=d.tables[0]
for r in info.rows:
    if r.cells[0].text.strip()=="Report date": set_cell(r.cells[1]._tc,"22 October 2026 (Green AI study added as Section 11; 3 October for Sections 1 to 10)")
    if r.cells[0].text.strip()=="Code and data":
        pass
# summary bullet
sb=para(T_SUMBUL,f"**A much greener system costs little accuracy (Section 11).** Measured on the GPU, the full system draws {cw:.1f} Wh to produce its five-fold predictions, {100*dl/cw:.0f}% of it to train the seven networks. Gradient boosting on ratios and language features, with no neural network, scores PR-AUC {f3(gb.pr)} against {f3(cur.pr)} (interval of the difference {sg(REP['lo'])} to {sg(REP['hi'])}) with no GPU work: about {cw/gw:.0f} times less GPU energy and CO₂e. If the network is kept, three seeds instead of seven save {sav('blend_finbert_fp16_k3'):.0f}% of its energy for {sg(k3['pr']-cur.pr)} PR-AUC.")
lastfound=find(lambda t:t.startswith('The auditor\'s report is the stronger text'))
lastfound.addnext(sb)
# conclusion: one added paragraph
concl=find(lambda t:t.startswith('Do the words in an Indian annual report'))
concl.addnext(para(T_BODY,f"The Green AI study adds a practical point: most of the energy goes into training the deep model, and gradient boosting on ratios and language features gives up {abs(REP['diff']):.3f} PR-AUC, inside the noise, for about {cw/gw:.0f} times less GPU energy. Where the energy cost matters, that model, or the blend with three seeds instead of seven, is the sensible deployment."))
# reproducibility table: add a row
rt=[t for t in d.tables if t.rows[0].cells[0].text=="File" and 'run_gpu.py' in t.rows[1].cells[0].text][0]
newr=copy.deepcopy(rt.rows[-1]._tr); tcs=newr.findall(W('w:tc'))
set_cell(tcs[0],"run_green.py, green_report.py, green_on_corrected_data.py, bppfinal/green.py, report/merge_green_section.py; results/green/ (in the code repository)")
set_cell(tcs[1],"The Green AI study of Section 11: energy-measured runs, the comparison tables, the Pareto figure, the selection rule and its decisions log (results/green/DECISIONS.md); green_on_corrected_data.py scores Table 22 from the saved predictions")
rt._tbl.append(newr)
d.save(OUT); print('saved',OUT)
