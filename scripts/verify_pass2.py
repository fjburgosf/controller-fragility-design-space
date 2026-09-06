"""PASADA 2 de verificacion. Texto, numeracion, referencias y coherencia.

Comprueba lo que la pasada 1 no toca. Que cada ecuacion, tabla y figura este
numerada, exista y se cite en orden. Que las citas del cuerpo correspondan con la
lista final. Que cada DOI resuelva en Crossref. Y que la Discusion y las
Conclusiones no afirmen nada que los Resultados no sostengan.
"""
from __future__ import annotations

import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "paper" / "manuscript_en.md").read_text(encoding="utf-8")
FIN = (ROOT / "paper" / "manuscript_en_ieee.md").read_text(encoding="utf-8")
REFS = {r["doi"].lower(): r for r in
        json.loads((ROOT / "paper" / "refs_verified.json").read_text(encoding="utf-8"))}

OK, MAL = [], []


def chk(nombre, bien, detalle):
    (OK if bien else MAL).append((nombre, detalle))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<46} {detalle}")


print("=" * 94)
print("A. ECUACIONES, TABLAS Y FIGURAS")
print("=" * 94)
eqs = [int(n) for n in re.findall(r"\\tag\{(\d+)\}", SRC)]
chk("ecuaciones numeradas sin salto", eqs == list(range(1, len(eqs) + 1)),
    f"{len(eqs)} ecuaciones, secuencia {eqs}")
for n in eqs:
    citada = re.search(rf"Eq\.\s*\({n}\)", SRC) is not None
    if not citada:
        chk(f"ecuacion {n} citada en el texto", False, "NO se cita")
chk("todas las ecuaciones citadas",
    all(re.search(rf"Eq\.\s*\({n}\)", SRC) for n in eqs),
    f"{sum(1 for n in eqs if re.search(rf'Eq\.\s*\({n}\)', SRC))} de {len(eqs)}")

for tipo, patron, carpeta, ext in (("Table", r"Table (\d+)", "tables/P2", None),
                                   ("Figure", r"Figure (\d+)", "figures/P2", ".png")):
    nums = sorted({int(n) for n in re.findall(patron, SRC)})
    chk(f"{tipo} numeracion consecutiva", nums == list(range(1, len(nums) + 1)),
        f"citadas {nums}")
    orden = [int(n) for n in re.findall(patron, SRC)]
    primera = {}
    for i, n in enumerate(orden):
        primera.setdefault(n, i)
    chk(f"{tipo} citadas en orden de aparicion",
        list(primera) == sorted(primera), f"orden de primera cita {list(primera)}")

fics = sorted((ROOT / "figures" / "P2").glob("*.png"))
chk("archivos de figura existen", len(fics) >= 3, f"{len(fics)} PNG en figures/P2")
tabs = sorted((ROOT / "tables" / "P2").glob("p2_tabla*.csv"))
chk("archivos de tabla existen", len(tabs) >= 6, f"{len(tabs)} CSV de tabla")

print()
print("=" * 94)
print("B. CITAS Y LISTA DE REFERENCIAS")
print("=" * 94)
marcadores = re.findall(r"\[\[([^\]]+)\]\]", SRC)
unicos = []
for d in marcadores:
    d = d.strip().lower()
    if d not in unicos:
        unicos.append(d)
chk("todo DOI citado esta en refs_verified", all(d in REFS for d in unicos),
    f"{sum(1 for d in unicos if d in REFS)} de {len(unicos)}")
chk("minimo 35 referencias citadas", len(unicos) >= 35, f"{len(unicos)} citadas")
rec = sum(1 for d in unicos if REFS[d]["year"] >= 2022)
chk("60 % o mas de los ultimos 4 anos", rec / len(unicos) >= 0.60,
    f"{rec} de {len(unicos)} = {100*rec/len(unicos):.0f} %")
chk("entre 15 y 50 referencias (Tecnura)", 15 <= len(unicos) <= 50, f"{len(unicos)}")

nums_cuerpo = [int(n) for n in re.findall(r"\[\[(\d+)\]\(#ref\d+\)\]", FIN)]
entradas = [int(n) for n in re.findall(r'\[\]\{#ref(\d+)\}', FIN)]
chk("numeracion del cuerpo empieza en 1 y es densa",
    sorted(set(nums_cuerpo)) == list(range(1, len(set(nums_cuerpo)) + 1)),
    f"{len(set(nums_cuerpo))} numeros distintos en el cuerpo")
chk("cada cita tiene entrada en la lista", set(nums_cuerpo) <= set(entradas),
    f"cuerpo {len(set(nums_cuerpo))}, lista {len(entradas)}")
chk("no hay entradas huerfanas en la lista", set(entradas) <= set(nums_cuerpo),
    f"huerfanas {sorted(set(entradas) - set(nums_cuerpo))}")
orden_ok = nums_cuerpo == sorted(set(nums_cuerpo), key=nums_cuerpo.index)
primera_ap = []
for n in nums_cuerpo:
    if n not in primera_ap:
        primera_ap.append(n)
chk("numeradas por orden de aparicion", primera_ap == list(range(1, len(primera_ap) + 1)),
    f"primeras apariciones {primera_ap[:12]}...")

print()
print("=" * 94)
print("C. RESOLUCION DE DOI EN CROSSREF")
print("=" * 94)
fallos_doi = []
for i, d in enumerate(unicos):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(d, safe="")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "P2-verify (mailto:fjburgosf@unal.edu.co)"})
        with urllib.request.urlopen(req, timeout=25) as r:
            m = json.load(r)["message"]
        t_cr = (m.get("title") or [""])[0].strip().lower()
        t_lo = REFS[d]["title"].strip().lower()
        if t_cr[:45] != t_lo[:45]:
            fallos_doi.append((d, "titulo distinto en Crossref"))
    except Exception as e:
        fallos_doi.append((d, f"no resuelve, {e.__class__.__name__}"))
    if (i + 1) % 12 == 0:
        print(f"      verificados {i+1} de {len(unicos)}", flush=True)
    time.sleep(0.12)
chk("los 36 DOI resuelven en Crossref", not fallos_doi,
    "todos resuelven" if not fallos_doi else f"{len(fallos_doi)} fallos")
for d, why in fallos_doi:
    print(f"        {d}  {why}")

print()
print("=" * 94)
print("D. COHERENCIA DE AFIRMACIONES")
print("=" * 94)
res = SRC.split("## 3. Results")[1].split("## 4. Discussion")[0]
dis = SRC.split("## 4. Discussion")[1].split("## 5. Conclusions")[0]
con = SRC.split("## 5. Conclusions")[1].split("## Acknowledgements")[0]

PROHIBIDAS = [
    ("regulador sin frontera", r"has no such boundary|carries no such boundary|regulator carries none|carries none"),
    ("cruce no observado afirmado como cruce", r"the (two )?(trends|curves) cross\b"),
    ("colapso fisico de la semilla", r"collapses? by a factor of twelve"),
    ("primer estudio", r"\bthe first (open|study|work|reproducible)"),
    ("lenguaje causal fuerte", r"\bproves that\b|\bdemonstrates conclusively\b"),
]
for nombre, pat in PROHIBIDAS:
    hit = re.search(pat, SRC, re.I)
    chk(f"sin {nombre}", hit is None, "ausente" if not hit else f"PRESENTE: {hit.group(0)}")

# El lenguaje experimental solo es sobrealcance en forma AFIRMATIVA. Declarar que
# NO hubo validacion experimental es obligatorio, y la negacion puede quedar varias
# palabras antes ("no experimental or hardware validation"), de modo que un
# lookbehind pegado a la palabra la marca como falso positivo. Se inspecciona una
# ventana previa, igual que en verify_numbering.
NEG_EXP = re.compile(r"\b(no|not|nor|without|absence of|neither)\b", re.I)
afirmativos = []
for _m in re.finditer(r"\b(?:experimental(?:ly)?|hardware|real.?time)\s+valida\w*", SRC, re.I):
    if not NEG_EXP.search(SRC[max(0, _m.start() - 90):_m.start()]):
        afirmativos.append(SRC[max(0, _m.start() - 60):_m.end() + 20].replace("\n", " "))
chk("sin lenguaje experimental afirmativo", not afirmativos,
    "ausente" if not afirmativos else f"PRESENTE: {afirmativos[0]}")

chk("las conclusiones declaran estudio computacional",
    "exclusively computational" in con, "declarado" if "exclusively computational" in con else "FALTA")
chk("existe seccion de limitaciones", "4.8 Limitations" in dis, "presente")
chk("declaracion de IA generativa presente",
    "generative artificial intelligence" in SRC, "presente")

# cada entrada admite variantes, porque el manuscrito escribe algunas
# proporciones en palabras y no en decimales
CIFRAS = [["0.725"], ["4.596"], ["224"], ["25", "Twenty five"], ["0.052"], ["0.368"],
          ["ninety six"], ["2.7"], ["11.378"], ["minus 8.2"], ["minus 21.2"],
          ["0.91", "91 percent"], ["0.33", "33 percent"]]
faltan = [v[0] for v in CIFRAS if not any(x in SRC for x in v)]
chk("cifras clave presentes en el texto", not faltan, "todas" if not faltan else f"faltan {faltan}")

print()
print("=" * 94)
print(f"RESUMEN PASADA 2   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 94)
for n, det in MAL:
    print(f"  DISCREPANCIA  {n}\n                {det}")
