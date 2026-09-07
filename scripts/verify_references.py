"""Verifica que CADA referencia citada existe y que sus metadatos son los reales.

Consulta Crossref para cada DOI citado en el manuscrito y compara titulo y ano
contra refs_verified.json. Una referencia inventada, un DOI mal copiado o unos
metadatos que se desviaron del registro aparecen aqui, no en la revision por pares.

Comprueba ademas que ningun titulo arrastre marcado JATS crudo, porque Crossref
devuelve algunos con etiquetas <i> o <sub> y saltos de linea, y sin limpiarlos la
entrada sale partida en la lista de referencias.

Requiere red. Uso: python scripts/verify_references.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
FUENTE = ROOT / "paper" / "manuscript_en.md"
FINAL = ROOT / "paper" / "manuscript_en_ieee.md"
ALMACEN = ROOT / "paper" / "refs_verified.json"
AGENTE = "verificacion-referencias/1.0 (mailto:fjburgosf@unal.edu.co)"

OK, MAL = [], []


def chk(nombre, bien, det):
    (OK if bien else MAL).append((nombre, det))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<58} {det}")


texto = FUENTE.read_text(encoding="utf-8")
citados = []
for d in re.findall(r"\[\[([^\]]+)\]\]", texto):
    if d not in citados:
        citados.append(d)
almacen = {r["doi"].lower(): r for r in json.loads(ALMACEN.read_text(encoding="utf-8"))}

print("=" * 100)
print(f"1. RESOLUCION EN CROSSREF DE LOS {len(citados)} DOI CITADOS")
print("=" * 100)
for i, d in enumerate(citados, 1):
    local = almacen.get(d.lower())
    if local is None:
        chk(f"{i:2}. {d}", False, "no esta en refs_verified.json")
        continue
    # una entrada sin DOI no vive en Crossref, se comprueba que traiga localizador
    if not d.startswith("10."):
        chk(f"{i:2}. {d[:40]}", bool(local.get("url")) and bool(local.get("accessed")),
            f"recurso en linea, {local.get('url', 'SIN URL')[:52]}")
        continue
    try:
        req = urllib.request.Request(f"https://api.crossref.org/works/{d}",
                                     headers={"User-Agent": AGENTE})
        with urllib.request.urlopen(req, timeout=25) as h:
            m = json.load(h)["message"]
    except Exception as e:
        chk(f"{i:2}. {d}", False, f"no resuelve: {str(e)[:44]}")
        time.sleep(0.15)
        continue

    def limpia(s):
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(s))).strip().lower()

    t_real, a_real = limpia(m["title"][0]), m.get("issued", {}).get("date-parts", [[None]])[0][0]
    coincide = limpia(local["title"])[:60] == t_real[:60] and local["year"] == a_real
    chk(f"{i:2}. {d[:40]}", coincide,
        f"{a_real}  {m['title'][0][:44]}" if coincide
        else f"local {local['year']} \"{local['title'][:34]}\" vs crossref {a_real} \"{m['title'][0][:34]}\"")
    time.sleep(0.15)

print()
print("=" * 100)
print("2. LA LISTA FINAL NO ARRASTRA MARCADO CRUDO")
print("=" * 100)
lista = FINAL.read_text(encoding="utf-8").split("## References")[1]
etiquetas = re.findall(r"<(?:i|b|sub|sup|em|strong|scp|mml:[a-z]+)[ >/]", lista)
chk("sin etiquetas HTML o JATS en las entradas", not etiquetas,
    "ninguna" if not etiquetas else f"{len(etiquetas)} etiquetas: {sorted(set(etiquetas))}")
partidas = [l for l in lista.splitlines() if l.startswith("[]{#ref")
            and "doi 10." not in l and "[Online]. Available" not in l]
chk("cada entrada cierra con doi o localizador en linea", not partidas,
    "todas" if not partidas else f"{len(partidas)} entradas truncadas")

print()
print("=" * 100)
print(f"RESUMEN REFERENCIAS   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
sys.exit(1 if MAL else 0)
