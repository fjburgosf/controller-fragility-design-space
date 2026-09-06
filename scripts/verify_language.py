"""Verifica que TODO lo que ve el lector esta en INGLES.

El manuscrito se escribe en ingles, y una tabla, un eje o una leyenda en español
es un defecto de envio que la conversion a docx no detecta y que al releer el
texto tampoco salta, porque el texto si esta en ingles. Aqui se revisan las dos
superficies que se generan por codigo.

1. Las tablas producidas en tables/P2, celda por celda y encabezado por encabezado.
2. Las cadenas visibles de los scripts de figuras, extraidas del arbol sintactico
   y no por expresion regular, de modo que solo se miran los argumentos que
   realmente llegan a un eje, un titulo, una leyenda o una anotacion.

Los comentarios y las cadenas de documentacion del codigo quedan fuera, porque no
los ve el lector del articulo.
"""
from __future__ import annotations

import ast
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_docx import TABLAS, FIGURAS

# funciones cuyo argumento de texto acaba dibujado en la figura
DIBUJAN = {"set_xlabel", "set_ylabel", "set_title", "suptitle", "text", "set_label",
           "annotate", "set_xticklabels", "set_yticklabels"}
CLAVES = {"label", "title", "xlabel", "ylabel"}

# marcadores de español. Palabras funcionales que el ingles no comparte, mas los
# diacriticos, que en un texto ingles solo aparecerian en un nombre propio.
PALABRAS = {
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "con", "sin", "por", "para", "que", "se", "su", "sus", "es", "son", "esta",
    "este", "esto", "estas", "estos", "al", "lo", "mas", "pero", "como", "cada",
    "tasa", "caida", "caidas", "muestreo", "horizonte", "guia", "guias", "espacio",
    "diseno", "digitos", "muro", "seguir", "peor", "mejor", "coste", "etapa",
    "cumple", "viola", "valor", "unidad", "magnitud", "masa", "carro", "pendulo",
    "longitud", "inercia", "limite", "fuerza", "voltio", "razon",
    "escalas", "temporales", "dominante", "rapido", "lento", "nivel",
    "controlador", "metodo", "ninguna", "ninguno", "formalizadas", "utiles",
}
SOLO_ES = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")


def normaliza(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def es_espanol(s: str):
    """Devuelve los indicios de español hallados en una cadena visible."""
    if SOLO_ES.search(s):
        return [f"diacritico {SOLO_ES.search(s).group(0)}"]
    # fuera la matematica en LaTeX, que no es ni ingles ni español
    limpio = re.sub(r"\$[^$]*\$", " ", s)
    tokens = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]{2,}", normaliza(limpio))
    hallados = sorted({t for t in tokens if t in PALABRAS})
    return hallados


OK, MAL = [], []


def chk(nombre, indicios, muestra=""):
    bien = not indicios
    (OK if bien else MAL).append((nombre, ", ".join(indicios) + (f"  en \"{muestra[:56]}\"" if muestra else "")))
    est = "ok " if bien else "MAL"
    det = "ingles" if bien else f"ESPAÑOL: {', '.join(indicios)}"
    print(f"  [{est}] {nombre:<52} {det}")


print("=" * 100)
print("1. TABLAS DEL MANUSCRITO")
print("=" * 100)
for num, (slug, _) in sorted(TABLAS.items()):
    df = pd.read_csv(ROOT / "tables" / "P2" / f"{slug}.csv")
    problemas, muestra = [], ""
    for celda in list(df.columns) + [str(v) for v in df.to_numpy().ravel()]:
        ind = es_espanol(str(celda))
        if ind:
            problemas += ind
            muestra = muestra or str(celda)
    chk(f"Tabla {num} ({slug})", sorted(set(problemas)), muestra)

print()
print("=" * 100)
print("2. TEXTO VISIBLE DE LAS FIGURAS")
print("=" * 100)
GENERA = {1: "fig_guidelines.py", 2: "fig_design_space.py", 3: "fig_design_space.py",
          4: "fig_rl_fragility.py", 5: "fig5_tails.py", 6: "fig6_timeseries.py"}
for num, script in sorted(GENERA.items()):
    arbol = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    visibles = []
    for n in ast.walk(arbol):
        if not isinstance(n, ast.Call):
            continue
        nombre = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        if nombre in DIBUJAN:
            visibles += [a for a in n.args]
        visibles += [k.value for k in n.keywords if k.arg in CLAVES]
    literales = []
    for v in visibles:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            literales.append(v.value)
        elif isinstance(v, ast.JoinedStr):     # f-string
            literales += [p.value for p in v.values
                          if isinstance(p, ast.Constant) and isinstance(p.value, str)]
    problemas, muestra = [], ""
    for s in literales:
        ind = es_espanol(s)
        if ind:
            problemas += ind
            muestra = muestra or s
    chk(f"Figura {num} ({script})", sorted(set(problemas)), muestra)

print()
print("=" * 100)
print("3. TABULADOR LITERAL POR \\tau EN CADENA NO CRUDA")
print("=" * 100)
# "$\tau$" sin la r inicial produce un tabulador. Es invisible al leer el archivo.
for carpeta, patron in ((ROOT / "tables" / "P2", "*.csv"), (ROOT / "tables" / "P2", "*.md")):
    for f in sorted(carpeta.glob(patron)):
        crudo = f.read_text(encoding="utf-8")
        chk(f"{f.name} sin tabulador incrustado", ["tabulador"] if "\t" in crudo else [])

print()
print("=" * 100)
print(f"RESUMEN IDIOMA   {len(OK)} verificadas   {len(MAL)} discrepancias")
print("=" * 100)
for n, d in MAL:
    print(f"  DISCREPANCIA  {n}\n                {d}")
sys.exit(1 if MAL else 0)
