"""Verificacion dedicada de NUMERACION, ORDEN DE APARICION y LLAMADO EN EL TEXTO.

Comprueba, para ecuaciones, tablas, figuras y referencias, que estan numeradas sin
saltos, que se citan al menos una vez en el cuerpo, y que el orden de la primera
cita coincide con el orden numerico. Tambien verifica que exista el archivo fisico
de cada figura y de cada tabla.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "paper" / "manuscript_en.md").read_text(encoding="utf-8")
FIN = (ROOT / "paper" / "manuscript_en_ieee.md").read_text(encoding="utf-8")

# el cuerpo excluye front matter y back matter
CUERPO = SRC.split("## 1. Introduction")[1].split("## Acknowledgements")[0]

OK, MAL = [], []


def chk(nombre, bien, det):
    (OK if bien else MAL).append((nombre, det))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<52} {det}")


def primera_aparicion(texto, patron):
    """Orden en que cada numero aparece por primera vez."""
    vistos = []
    for m in re.finditer(patron, texto):
        n = int(m.group(1))
        if n not in vistos:
            vistos.append(n)
    return vistos


print("=" * 100)
print("1. ECUACIONES")
print("=" * 100)
definidas = [int(n) for n in re.findall(r"\\tag\{(\d+)\}", SRC)]
chk("definidas y numeradas sin saltos", definidas == list(range(1, len(definidas) + 1)),
    f"{len(definidas)} ecuaciones, secuencia {definidas}")
chk("aparecen en el documento en orden ascendente", definidas == sorted(definidas),
    f"orden de definicion {definidas}")
citadas = primera_aparicion(CUERPO, r"Eq\.\s*\((\d+)\)")
sin_citar = [n for n in definidas if n not in citadas]
chk("todas citadas en el cuerpo", not sin_citar,
    "todas" if not sin_citar else f"SIN CITAR {sin_citar}")
chk("orden de primera cita ascendente", citadas == sorted(citadas),
    f"orden de primera cita {citadas}")
huerfanas = [n for n in citadas if n not in definidas]
chk("no se cita ninguna ecuacion inexistente", not huerfanas,
    "ninguna" if not huerfanas else f"citadas pero no definidas {huerfanas}")

print()
print("=" * 100)
print("2. TABLAS")
print("=" * 100)
tab_cit = primera_aparicion(CUERPO, r"Table (\d+)")
chk("numeracion consecutiva desde 1", sorted(tab_cit) == list(range(1, len(tab_cit) + 1)),
    f"citadas {sorted(tab_cit)}")
chk("orden de primera cita ascendente", tab_cit == sorted(tab_cit),
    f"orden de primera cita {tab_cit}")
archivos = sorted((ROOT / "tables" / "P2").glob("p2_tabla*.csv"))
nums_arch = sorted(int(re.search(r"tabla(\d+)", f.name).group(1)) for f in archivos)
chk("existe un archivo por cada tabla citada", set(tab_cit) <= set(nums_arch),
    f"archivos {nums_arch}, citadas {sorted(tab_cit)}")
chk("no hay tablas generadas que el texto no cite", set(nums_arch) <= set(tab_cit),
    "ninguna sobra" if set(nums_arch) <= set(tab_cit)
    else f"generadas sin citar {sorted(set(nums_arch)-set(tab_cit))}")

print()
print("=" * 100)
print("3. FIGURAS")
print("=" * 100)
fig_cit = primera_aparicion(CUERPO, r"Figure (\d+)")
chk("numeracion consecutiva desde 1", sorted(fig_cit) == list(range(1, len(fig_cit) + 1)),
    f"citadas {sorted(fig_cit)}")
chk("orden de primera cita ascendente", fig_cit == sorted(fig_cit),
    f"orden de primera cita {fig_cit}")
# fuente unica de verdad: el mismo mapa que usa el constructor del docx
sys.path.insert(0, str(ROOT / "scripts"))
from build_docx import FIGURAS

usadas = [ROOT / "figures" / "P2" / a for _, (a, _) in sorted(FIGURAS.items())]
nums_fig = sorted(FIGURAS)
chk("existe archivo por cada figura citada", set(fig_cit) <= set(nums_fig),
    f"registradas en el constructor {nums_fig}, citadas {sorted(fig_cit)}")
chk("el constructor no registra figuras que el texto no cite", set(nums_fig) <= set(fig_cit),
    "ninguna sobra" if set(nums_fig) <= set(fig_cit)
    else f"registradas sin citar {sorted(set(nums_fig)-set(fig_cit))}")
sueltas = sorted(f.name for f in (ROOT / "figures" / "P2").glob("fig*.png")
                 if f.name not in {a for _, (a, _) in FIGURAS.items()})
chk("no hay figuras huerfanas en la carpeta activa", not sueltas,
    "ninguna" if not sueltas else f"sin usar {sueltas}")
for f in usadas:
    chk(f"archivo {f.name} no vacio", f.stat().st_size > 5000, f"{f.stat().st_size} bytes")

print()
print("=" * 100)
print("4. REFERENCIAS")
print("=" * 100)
cuerpo_fin = FIN.split("## 1. Introduction")[1].split("## References")[0]
ref_cit = primera_aparicion(cuerpo_fin, r"\[\[(\d+)\]\(#ref\d+\)\]")
entradas = [int(n) for n in re.findall(r'\[\]\{#ref(\d+)\}', FIN)]
chk("numeradas por orden de aparicion en el texto",
    ref_cit == list(range(1, len(ref_cit) + 1)),
    f"primeras apariciones {ref_cit[:10]} ... {ref_cit[-3:]}")
chk("lista final consecutiva desde 1", entradas == list(range(1, len(entradas) + 1)),
    f"{len(entradas)} entradas")
chk("cada cita tiene entrada", set(ref_cit) <= set(entradas),
    f"citadas {len(set(ref_cit))}, entradas {len(entradas)}")
chk("ninguna entrada sin citar", set(entradas) <= set(ref_cit),
    "ninguna" if set(entradas) <= set(ref_cit) else f"{sorted(set(entradas)-set(ref_cit))}")
enlaces = len(re.findall(r"\[\[\d+\]\(#ref\d+\)\]", FIN))
anclas = len(re.findall(r'\[\]\{#ref\d+\}', FIN))
chk("toda cita esta hipervinculada", enlaces >= len(set(ref_cit)),
    f"{enlaces} enlaces en el cuerpo, {anclas} anclas en la lista")

print()
print("=" * 100)
print("5. LENGUAJE DE VALIDACION (solo forma afirmativa)")
print("=" * 100)
NEG = re.compile(r"\b(no|not|nor|without|absence of)\b", re.I)
sospechas = []
for m in re.finditer(r"\b(?:experimental(?:ly)?|hardware|real.?time)\s+valida\w*", SRC, re.I):
    ventana = SRC[max(0, m.start() - 80):m.start()]
    if not NEG.search(ventana):
        sospechas.append(SRC[max(0, m.start() - 60):m.end() + 20].replace("\n", " "))
chk("sin afirmacion de validacion experimental", not sospechas,
    "ninguna" if not sospechas else f"{len(sospechas)} casos")
for s in sospechas:
    print(f"        ...{s}...")

print()
print("=" * 100)
print(f"RESUMEN NUMERACION   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
