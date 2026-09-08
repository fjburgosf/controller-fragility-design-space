"""Verificacion COMPLETA de la reconstruccion del docx contra sus fuentes.

El docx es el archivo que ve la revista, y la conversion puede perder cosas en
silencio. Aqui se abre el archivo producido y se comprueba, contra el manuscrito
fuente y contra los CSV de las tablas, que todo llego y llego en orden.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_docx import TABLAS, FIGURAS

import os
DOCX = ROOT / "paper" / os.environ.get("P2_DOCX", "P2_manuscrito.docx")
SRC = (ROOT / "paper" / "manuscript_en_ieee.md").read_text(encoding="utf-8")
FUENTE = (ROOT / "paper" / "manuscript_en.md").read_text(encoding="utf-8")
NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

OK, MAL = [], []


def chk(nombre, bien, det):
    (OK if bien else MAL).append((nombre, det))
    print(f"  [{'ok ' if bien else 'MAL'}] {nombre:<54} {det}")


doc = Document(str(DOCX))
xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")


def bloques(padre):
    """Parrafos y tablas en el orden real del documento."""
    for hijo in padre.element.body.iterchildren():
        if hijo.tag == f"{NS}p":
            yield Paragraph(hijo, padre)
        elif hijo.tag == f"{NS}tbl":
            yield Table(hijo, padre)


ORDEN = list(bloques(doc))
PARRAFOS = [b for b in ORDEN if isinstance(b, Paragraph)]
TABLAS_DOC = [b for b in ORDEN if isinstance(b, Table)]
TEXTO = "\n".join(p.text for p in PARRAFOS)

print("=" * 100)
print("1. INTEGRIDAD DEL ARCHIVO")
print("=" * 100)
chk("el docx abre y tiene contenido", len(PARRAFOS) > 100, f"{len(PARRAFOS)} parrafos")
chk("sin residuo de LaTeX", "tag{" not in xml and "$$" not in xml,
    f"tag {xml.count('tag{')}, dollar {xml.count('$$')}")
chk("sin marcadores de cita sin resolver", "[[" not in TEXTO, "ninguno")
pendientes = re.findall(r"\[[A-ZÁÉÍÓÚ ]*PENDIENTE[^\]]*\]", TEXTO)
ESPERADOS = 6   # afiliacion, ORCID y CvLAC de los dos coautores pendientes
chk("los unicos pendientes son los datos de coautores", len(pendientes) == ESPERADOS,
    f"{len(pendientes)} marcadores, se esperan {ESPERADOS}")

print()
print("=" * 100)
print("2. SECCIONES, EN ORDEN")
print("=" * 100)
# La declaracion de IA debe ser el ultimo apartado antes de References, que es
# como lo exige Tecnura.
ENCABEZADOS = ["Abstract", "Resumen", "1. Introduction", "2. Methodology", "3. Results",
               "4. Discussion", "5. Conclusions", "Acknowledgements", "Data availability",
               "Conflict of interest", "Funding", "CRediT author statement",
               "Declaration on the use of generative artificial intelligence", "References"]
pos, desordenados, ausentes = -1, [], []
for h in ENCABEZADOS:
    idx = next((i for i, p in enumerate(PARRAFOS)
                if p.style.name.startswith("Heading") and p.text.strip() == h), None)
    if idx is None:
        ausentes.append(h)
    elif idx < pos:
        desordenados.append(h)
    else:
        pos = idx
chk("todas las secciones presentes", not ausentes,
    f"las {len(ENCABEZADOS)}" if not ausentes else f"faltan {ausentes}")
chk("las secciones aparecen en orden", not desordenados,
    "en orden" if not desordenados else f"fuera de sitio {desordenados}")

print()
print("=" * 100)
print("3. FIGURAS")
print("=" * 100)
blips = re.findall(r"<a:blip", xml)
chk("una imagen incrustada por figura", len(blips) == len(FIGURAS),
    f"{len(blips)} imagenes, {len(FIGURAS)} figuras registradas")
leyendas = [i for i, p in enumerate(PARRAFOS) if re.match(r"\s*Figure \d+\.", p.text)]
nums_ley = [int(re.search(r"Figure (\d+)", PARRAFOS[i].text).group(1)) for i in leyendas]
chk("una leyenda por figura, numeradas en orden", nums_ley == sorted(FIGURAS),
    f"leyendas {nums_ley}")
for n in sorted(FIGURAS):
    es_leyenda = lambda s: re.match(rf"\s*Figure {n}\.", s) is not None
    cita = next((i for i, p in enumerate(PARRAFOS)
                 if f"Figure {n}" in p.text and not es_leyenda(p.text)), None)
    ley = next((i for i in leyendas if f"Figure {n}." in PARRAFOS[i].text), None)
    chk(f"Figura {n} citada antes de su leyenda",
        cita is not None and ley is not None and cita < ley,
        f"cita en parrafo {cita}, leyenda en {ley}")

print()
print("=" * 100)
print("4. TABLAS, CONTRASTADAS CONTRA SU CSV")
print("=" * 100)
# los envoltorios de ecuacion son tablas de dos columnas y a lo sumo dos filas
CONTENIDO = [t for t in TABLAS_DOC if not (len(t.rows) <= 2 and len(t.columns) == 2)]
ECUACION = [t for t in TABLAS_DOC if len(t.rows) <= 2 and len(t.columns) == 2]
chk("una tabla nativa por tabla del manuscrito", len(CONTENIDO) == len(TABLAS),
    f"{len(CONTENIDO)} tablas de contenido, {len(TABLAS)} esperadas")
for (num, (slug, _)), tab in zip(sorted(TABLAS.items()), CONTENIDO):
    csv = pd.read_csv(ROOT / "tables" / "P2" / f"{slug}.csv")
    filas_doc = len(tab.rows) - 1          # descontar el encabezado
    chk(f"Tabla {num} conserva todas sus filas", filas_doc == len(csv),
        f"docx {filas_doc}, csv {len(csv)}")
    celdas = {c.text.strip() for f in tab.rows for c in f.cells}
    primera = re.sub(r"\$[^$]*\$", "", str(csv.iloc[0, 0])).strip()
    hallada = any(primera and primera in c for c in celdas)
    chk(f"Tabla {num} conserva su primer valor", hallada, primera[:36])

print()
print("=" * 100)
print("5. ECUACIONES")
print("=" * 100)
n_eq = len(re.findall(r"\\tag\{(\d+)\}", FUENTE))
omml = len(re.findall(r"<m:oMath[ >]", xml))
chk("cada ecuacion numerada tiene su envoltorio", len(ECUACION) == n_eq,
    f"{len(ECUACION)} envoltorios, {n_eq} ecuaciones en la fuente")
chk("hay matematica nativa OMML, no imagenes ni texto plano", omml >= n_eq,
    f"{omml} bloques OMML")
celdas_eq = [c.text.strip() for tb in ECUACION for f in tb.rows for c in f.cells]
nums_eq = sorted(int(m.group(1)) for m in
                 (re.fullmatch(r"\((\d+)\)", c) for c in celdas_eq) if m)
chk("los numeros de ecuacion sobreviven, sin saltos", nums_eq == list(range(1, n_eq + 1)),
    f"numeros hallados {nums_eq}")
chk("ninguna variable quedo como texto entre dolares", "$" not in TEXTO,
    "ninguna" if "$" not in TEXTO else f"{TEXTO.count('$')} apariciones")
# Word dibuja como recuadro de marcador de posicion cualquier corrida OMML que
# solo contenga un espacio o este vacia. Es lo que producen \, \; \quad y \ .
_omml_boxes = re.findall(r"<m:t/>|<m:t>\s*</m:t>", xml)
chk("ninguna ecuacion tiene recuadro vacio (run OMML solo-espacio)", not _omml_boxes,
    "ninguno" if not _omml_boxes else f"{len(_omml_boxes)} corridas vacias")

print()
print("=" * 100)
print("6. REFERENCIAS Y VINCULOS")
print("=" * 100)
entradas = re.findall(r"^\[(\d+)\]\s", TEXTO, re.M)
n_ref = len(re.findall(r"\[\]\{#ref(\d+)\}", SRC))
chk("la lista final conserva todas las entradas", len(entradas) == n_ref,
    f"docx {len(entradas)}, fuente {n_ref}")
chk("entradas numeradas sin saltos",
    [int(x) for x in entradas] == list(range(1, len(entradas) + 1)), f"{len(entradas)} entradas")
n_link = len(re.findall(r"<w:hyperlink", xml))
chk("las citas del cuerpo son hipervinculos", n_link >= n_ref, f"{n_link} hipervinculos")
anclas = set(re.findall(r'<w:bookmarkStart[^>]*w:name="(ref\d+)"', xml))
destinos = set(re.findall(r'<w:hyperlink[^>]*w:anchor="(ref\d+)"', xml))
rotos = destinos - anclas
chk("ningun vinculo apunta a un ancla inexistente", not rotos,
    f"{len(anclas)} anclas, {len(destinos)} destinos"
    + ("" if not rotos else f", rotos {sorted(rotos)[:5]}"))

print()
print("=" * 100)
print("7. NADA SE PERDIO EN LA CONVERSION")
print("=" * 100)


def limpia(t):
    t = re.sub(r"\[\[(\d+)\]\(#ref\d+\)\]", r"[\1]", t)
    t = re.sub(r"\\tag\{\d+\}", " ", t)
    t = re.sub(r"[*_`#|>$]", " ", t)
    return re.sub(r"\s+", " ", t)


pal_src = len(limpia(SRC).split())
pal_doc = len(limpia(TEXTO).split())
# el docx suma leyendas y cuerpos de tabla, asi que solo se exige que no falte texto
chk("el docx no perdio texto de la fuente", pal_doc >= pal_src * 0.95,
    f"fuente {pal_src} palabras, docx {pal_doc}")
PLANO = re.sub(r"\s+", " ", TEXTO)
for frase in ("ninety six configurations that satisfy the horizon rule",
              "twenty five violating both rules survived every realisation",
              "11.378 ms in its slowest step",
              "2.079 m, about seven times that half length",
              "no experimental or hardware validation was performed"):
    clave = re.sub(r"\s+", " ", frase)
    chk(f"llega intacta: {clave[:38]}", clave in PLANO, "presente" if clave in PLANO else "AUSENTE")

print()
print("=" * 100)
print(f"RESUMEN DOCX   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
