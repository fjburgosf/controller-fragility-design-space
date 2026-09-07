"""Construye el docx de Tecnura desde el manuscrito, con tablas y ecuaciones numeradas.

Resuelve dos cosas que la conversion directa pierde.

1. Las tablas se citan en el texto pero viven en tables/P2. Aqui se insertan como
   tablas de Markdown justo despues del parrafo que las cita por primera vez, de
   modo que pandoc las convierte en tablas nativas de Word.

2. Word no entiende \\tag de LaTeX, asi que el numero de ecuacion se pierde. Se
   sustituye por una tabla de una fila y dos columnas sin bordes, con la ecuacion
   centrada y el numero alineado a la derecha entre parentesis, que es lo que pide
   la revista. La ecuacion sigue siendo OMML nativo.

Uso. python scripts/build_docx.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PANDOC = Path(r"C:/Users/UNAL/AppData/Local/Pandoc/pandoc.exe")
TAB = ROOT / "tables" / "P2"
SRC = ROOT / "paper" / "manuscript_en_ieee.md"
TMP = ROOT / "paper" / "_manuscrito_docx.md"
import os
OUT = ROOT / "paper" / os.environ.get("P2_DOCX", "P2_manuscrito.docx")

# tabla del manuscrito -> archivo generado, y su leyenda
TABLAS = {
    1: ("p2_tabla1_planta", "Plant parameters and closed loop modal structure."),
    2: ("p2_tabla2_guias", "Contrast of the two tuning rules over the full design space."),
    3: ("p2_tabla3_jerarquia", "Ranking of the design variables of the predictive controller."),
    4: ("p2_tabla4_interaccion", "Interaction between prediction horizon and terminal weight."),
    5: ("p2_tabla5_comparacion_final", "Final comparison of the three controller families."),
    6: ("p2_tabla6_coste", "Computational cost per control step and design freedom exposed."),
}
FIGURAS = {
    1: ("fig1_guidelines.png",
        "Falsification of the two tuning rules over the complete grid of 224 predictive controller configurations."),
    2: ("fig2_map_riccati.png",
        "Design space with the Riccati terminal weight. Cell values are the fraction of "
        "realisations in which the pendulum was lost."),
    3: ("fig3_map_stage.png",
        "Design space with the stage terminal weight, on the same scale as Figure 2."),
    4: ("fig4_rl_fragility.png",
        ("Fragility of learned control across configuration, seeds and the training "
        "boundary. Evaluation return is a sum of negated stage costs, so it is bounded "
        "above by zero and a value nearer zero is better.")),
    5: ("fig5_tails.png",
        "Distribution of angular error by controller family across the training boundary."),
    6: ("fig6_timeseries.png",
        ("Closed loop response of the four controllers on one shared realisation. "
        "The angular error is wrapped to the interval from minus pi to pi, so a jump "
        "between plus and minus three radians is the pendulum passing through the "
        "hanging position rather than a discontinuity in the signal.")),
}


def cuerpo_tabla(slug: str) -> str:
    """Devuelve solo la tabla Markdown del archivo generado, sin la leyenda."""
    md = (TAB / f"{slug}.md").read_text(encoding="utf-8")
    return "\n".join(l for l in md.split("\n") if l.strip().startswith("|")).strip()


def main() -> int:
    t = SRC.read_text(encoding="utf-8")

    # ---- ecuaciones. \tag{n} -> tabla sin bordes con el numero a la derecha ----
    # El microespaciado de LaTeX (\, \; \: \! \quad \qquad y \ ) se convierte en
    # corridas OMML que solo contienen un espacio, y Word las dibuja como el
    # recuadro vacio del marcador de posicion. Se quita del cuerpo de cada
    # ecuacion antes de pasar por pandoc. Es puramente tipografico y OMML aplica
    # su propio espaciado entre simbolos.
    ESPACIO_TIPOGRAFICO = re.compile(r"\\[,;:!]|\\q?quad\b|\\(?=\s)")

    def eq(m):
        cuerpo, n = m.group(1).strip(), m.group(2)
        cuerpo = ESPACIO_TIPOGRAFICO.sub(" ", cuerpo)
        return (f"\n|  |  |\n|:--:|--:|\n| ${cuerpo}$ | ({n}) |\n")
    t, n_eq = re.subn(r"\$\$\s*(.*?)\s*\\tag\{(\d+)\}\s*\$\$", eq, t, flags=re.S)

    # ---- tablas. insertar tras el parrafo que las cita por primera vez ----
    n_tab = 0
    for num, (slug, leyenda) in sorted(TABLAS.items(), reverse=True):
        if not (TAB / f"{slug}.md").exists():
            print(f"  aviso, falta {slug}.md"); continue
        m = next((x for x in re.finditer(rf"(^.*?\bTable {num}\b.*?$)", t, re.M)
                  if not x.group(0).lstrip().startswith("**Table")), None)
        if not m:
            print(f"  aviso, Table {num} no se cita"); continue
        bloque = f"\n\n**Table {num}.** {leyenda}\n\n{cuerpo_tabla(slug)}\n"
        fin = m.end()
        while fin < len(t) and t[fin:fin + 2] != "\n\n":
            fin += 1
        t = t[:fin] + bloque + t[fin:]
        n_tab += 1

    # ---- figuras. insertar tras el parrafo que las cita ----
    n_fig = 0
    for num, (arch, leyenda) in sorted(FIGURAS.items(), reverse=True):
        if not (ROOT / "figures" / "P2" / arch).exists():
            print(f"  aviso, falta {arch}"); continue
        m = next((x for x in re.finditer(rf"(^.*?\bFigure {num}\b.*?$)", t, re.M)
                  if not x.group(0).lstrip().startswith("**Figure")), None)
        if not m:
            print(f"  aviso, Figure {num} no se cita"); continue
        fin = m.end()
        while fin < len(t) and t[fin:fin + 2] != "\n\n":
            fin += 1
        t = t[:fin] + f"\n\n![**Figure {num}.** {leyenda}](../figures/P2/{arch})\n" + t[fin:]
        n_fig += 1

    TMP.write_text(t, encoding="utf-8")
    cmd = [str(PANDOC), str(TMP), "-o", str(OUT),
           "--from", "markdown+tex_math_dollars+pipe_tables+yaml_metadata_block+raw_html",
           "--to", "docx", "--metadata", "lang=en", "--resource-path", str(ROOT / "paper")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("pandoc fallo\n", r.stderr[:900]); return 1

    z = zipfile.ZipFile(OUT); x = z.read("word/document.xml").decode("utf-8")
    print(f"\n{OUT.name}  {OUT.stat().st_size} bytes")
    print(f"  ecuaciones insertadas    {n_eq}")
    print(f"  tablas insertadas        {n_tab}")
    print(f"  figuras insertadas       {n_fig}")
    print("  --- verificacion del docx ---")
    print(f"  OMML <m:oMath>           {len(re.findall('<m:oMath[ >]', x))}")
    print(f"  tablas nativas <w:tbl>   {len(re.findall('<w:tbl>', x))}")
    print(f"  imagenes <a:blip>        {len(re.findall('<a:blip', x))}")
    print(f"  hipervinculos            {len(re.findall('<w:hyperlink', x))}")
    print(f"  restos de LaTeX sin convertir  tag {x.count('tag{')}  dollar {x.count('$$')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
