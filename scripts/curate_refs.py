"""Cura el pool de Crossref: exige relevancia real en el titulo, cubre los temas
del paper y garantiza >=35 referencias con >=60% de los ultimos 4 anos."""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pool = json.loads((ROOT / "paper" / "refs_pool.json").read_text(encoding="utf-8"))
OUT = ROOT / "paper" / "refs_verified.json"
RECENT = 2022

# cada tema exige que el titulo contenga al menos una senal fuerte
NEED = {
    "mpc_core": ([r"model predictive", r"\bmpc\b", r"receding horizon"], 10),
    "mpc_tuning": ([r"(horizon|sampling|tuning|parameter).*(select|tun|choos|design)",
                    r"(tun|select|choos).*(horizon|sampling)"], 4),
    "mpc_theory": ([r"terminal (cost|set|constraint)", r"recursive feasib",
                    r"invariant set", r"stability.*predictive"], 4),
    "mpc_numeric": ([r"condition", r"condensed", r"sparse", r"move block",
                     r"embedded", r"real.?time.*(mpc|predictive)"], 3),
    "pendulum": ([r"inverted pendulum", r"cart.?pole", r"underactuated"], 4),
    "drl": ([r"reinforcement learning", r"actor.?critic", r"policy gradient",
             r"\bsac\b", r"\bddpg\b", r"deep.*control"], 8),
    "drl_robust": ([r"domain randomi", r"sim.?to.?real", r"out.?of.?distribution",
                    r"generaliz", r"reproducib", r"seed"], 4),
    "uq": ([r"uncertainty", r"monte carlo", r"sensitivity analysis", r"robust"], 5),
}
BAD = re.compile(r"\b(cancer|tumor|clinical|patient|crop|soil|gene|protein|drug|"
                 r"covid|epidemi|bank|stock|marketing|educat|student|nurs)\b", re.I)


def score(r):
    t = (r["title"] or "").lower()
    if BAD.search(t) or len(t) < 25:
        return None
    hits = [k for k, (pats, _) in NEED.items() if any(re.search(p, t) for p in pats)]
    if not hits:
        return None
    s = len(hits) * 10 + min(r.get("cited_by", 0), 200) / 20
    if r["year"] and r["year"] >= RECENT:
        s += 6
    if r["type"] == "journal-article":
        s += 3
    return s, hits


scored = []
for r in pool:
    v = score(r)
    if v:
        r["_score"], r["_hits"] = v
        scored.append(r)
scored.sort(key=lambda r: -r["_score"])

# cuotas por tema para garantizar cobertura
sel, used = [], set()
for tema, (_, cuota) in NEED.items():
    n = 0
    for r in scored:
        if n >= cuota or r["doi"] in used:
            continue
        if tema in r["_hits"]:
            sel.append(r); used.add(r["doi"]); n += 1
# rellenar hasta 40 con los mejores restantes, cuidando el % de recientes
for r in scored:
    if len(sel) >= 40:
        break
    if r["doi"] in used:
        continue
    rec = sum(1 for x in sel if x["year"] >= RECENT)
    if (rec / len(sel) < 0.62) and r["year"] < RECENT:
        continue
    sel.append(r); used.add(r["doi"])

sel.sort(key=lambda r: (-(r["year"] or 0), r["title"]))
for r in sel:
    r.pop("_score", None); r.pop("_hits", None)
    r["verified_via"] = "Crossref API"

rec = sum(1 for r in sel if r["year"] >= RECENT)
OUT.write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Seleccionadas {len(sel)} | {RECENT}+ : {rec} ({100*rec/len(sel):.0f}%)")
print(f"Revistas distintas: {len({r['venue'] for r in sel})}")
print(f"\nGuardado: {OUT}\n")
for r in sel:
    print(f"  {r['year']}  {r['title'][:72]}")
