"""Recolecta referencias VERIFICADAS desde la API de Crossref (P2).

Cada entrada proviene directamente de Crossref, por lo que su DOI y metadata
estan verificados por construccion. No se escribe ninguna referencia de memoria.

Salida: paper/refs_pool.json  (pool de candidatas para curar)
"""
from __future__ import annotations
import json, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "refs_pool.json"
MAILTO = "fjburgosf@unal.edu.co"

TOPICS = {
    "mpc_tuning": "model predictive control tuning prediction horizon selection guidelines",
    "mpc_sampling": "model predictive control sampling time selection closed loop bandwidth",
    "mpc_terminal": "model predictive control terminal cost terminal set stability guarantee",
    "mpc_feasibility": "model predictive control recursive feasibility control invariant set",
    "mpc_blocking": "move blocking model predictive control input parameterization",
    "mpc_condensed": "condensed sparse formulation model predictive control numerical conditioning",
    "mpc_embedded": "embedded real time model predictive control microcontroller implementation",
    "mpc_unstable": "model predictive control unstable systems prediction horizon ill conditioning",
    "cartpole": "inverted pendulum cart control benchmark stabilization",
    "lqr_mpc": "comparison linear quadratic regulator model predictive control performance",
    "drl_control": "deep reinforcement learning continuous control soft actor critic",
    "ddpg": "deep deterministic policy gradient continuous control benchmark",
    "domain_rand": "domain randomization sim to real transfer reinforcement learning robustness",
    "rl_robust": "robustness deep reinforcement learning policies out of distribution generalization",
    "rl_reproducibility": "reproducibility deep reinforcement learning seed variance evaluation",
    "rl_vs_classic": "reinforcement learning versus classical control comparison benchmark",
    "uq_control": "uncertainty quantification Monte Carlo simulation control systems",
    "sensitivity": "global sensitivity analysis model parameters variance based",
    "robust_control": "robust control parametric uncertainty performance degradation",
    "tail_risk": "worst case tail risk analysis control performance distribution",
}

TYPES = {"journal-article", "proceedings-article", "book-chapter"}


def fetch(query: str, rows: int = 22, from_year: int | None = None):
    p = {"query.bibliographic": query, "rows": rows, "mailto": MAILTO,
         "select": "DOI,title,author,issued,container-title,type,is-referenced-by-count"}
    if from_year:
        p["filter"] = f"from-pub-date:{from_year}-01-01"
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={"User-Agent": f"P2-refs (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)["message"]["items"]


def norm(it, topic):
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    au = it.get("author") or []
    return {
        "doi": it.get("DOI"),
        "title": (it.get("title") or [""])[0].strip(),
        "year": yr,
        "authors": [f"{a.get('family','')} {a.get('given','')}".strip() for a in au[:6]],
        "n_authors": len(au),
        "venue": (it.get("container-title") or [""])[0],
        "type": it.get("type"),
        "cited_by": it.get("is-referenced-by-count", 0),
        "topic": topic,
    }


pool: dict[str, dict] = {}
for topic, q in TOPICS.items():
    for from_year in (2022, None):          # primero recientes, luego historicas
        try:
            items = fetch(q, from_year=from_year)
        except Exception as e:
            print(f"  {topic} ({from_year}): fallo {e.__class__.__name__}", flush=True)
            continue
        add = 0
        for it in items:
            r = norm(it, topic)
            if not r["doi"] or not r["title"] or r["type"] not in TYPES:
                continue
            if not r["year"] or r["year"] < 2000 or r["year"] > 2026:
                continue
            if r["doi"] not in pool:
                pool[r["doi"]] = r
                add += 1
        print(f"  {topic:22} desde={from_year or 'todo':>4}  +{add:2d}  (pool {len(pool)})", flush=True)
        time.sleep(0.6)

OUT.write_text(json.dumps(list(pool.values()), indent=1, ensure_ascii=False), encoding="utf-8")
rec = sum(1 for r in pool.values() if r["year"] and r["year"] >= 2022)
print(f"\nPool total {len(pool)} | 2022+ {rec} ({100*rec/max(len(pool),1):.0f}%)")
print(f"Guardado: {OUT}")
