"""Convierte marcadores [[doi]] en citas IEEE numeradas por orden de aparicion.

Genera el manuscrito final con cada cita vinculada a su entrada en la lista, y
la lista de referencias en formato IEEE. Soporta tambien estilo APA para revistas
que lo exijan (seccion 25.10 de AGENTS.md).

Uso
    python scripts/build_references.py paper/manuscript_en.md [--estilo ieee|apa]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent


def apellido(a: str) -> str:
    return a.split()[0] if a else ""


def iniciales(a: str) -> str:
    partes = a.split()
    return " ".join(p[0] + "." for p in partes[1:]) if len(partes) > 1 else ""


def ieee(r, n):
    au = r["authors"]
    if not au:
        autores = ""
    elif len(au) == 1:
        autores = f"{iniciales(au[0])} {apellido(au[0])}"
    elif r["n_authors"] > 6:
        autores = f"{iniciales(au[0])} {apellido(au[0])} et al."
    else:
        partes = [f"{iniciales(a)} {apellido(a)}" for a in au]
        autores = ", ".join(partes[:-1]) + " and " + partes[-1]
    ven = f", *{r['venue']}*" if r["venue"] else ""
    # Una entrada sin DOI se cierra con su localizador y la fecha de consulta, que
    # es lo que pide IEEE para un recurso en linea. La documentacion de una
    # herramienta comercial no tiene DOI y aun asi es la guia que un disenador
    # consulta de verdad, de modo que debe poder citarse.
    if r.get("url") and not str(r.get("doi", "")).startswith("10."):
        acc = f" [Accessed {r['accessed']}]" if r.get("accessed") else ""
        # sin autor personal, la organizacion emisora ocupa el lugar del autor y no
        # se repite como publicacion, que es la forma IEEE para documentacion
        cab = autores if autores else r["venue"]
        med = ven if autores else ""
        return f"[{n}] {cab}, {r['title']}{med}, {r['year']}. [Online]. Available {r['url']}{acc}"
    return f"[{n}] {autores}, {r['title']}{ven}, {r['year']}. doi {r['doi']}"


def apa(r):
    au = r["authors"]
    if not au:
        autores = ""
    elif len(au) == 1:
        autores = f"{apellido(au[0])}, {iniciales(au[0])}"
    else:
        partes = [f"{apellido(a)}, {iniciales(a)}" for a in au]
        autores = ", ".join(partes[:-1]) + " & " + partes[-1]
    ven = f" *{r['venue']}*." if r["venue"] else ""
    return f"{autores} ({r['year']}). {r['title']}.{ven} https://doi.org/{r['doi']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ruta")
    ap.add_argument("--estilo", choices=["ieee", "apa"], default="ieee")
    a = ap.parse_args()

    src = Path(a.ruta)
    texto = src.read_text(encoding="utf-8")
    refs = {r["doi"].lower(): r for r in
            json.loads((ROOT / "paper" / "refs_verified.json").read_text(encoding="utf-8"))}
    # Crossref devuelve algunos titulos con marcado JATS crudo, etiquetas como
    # <i> y <sub> y saltos de linea con sangria. Sin limpiarlos, la entrada sale
    # partida en la lista de referencias y el lector ve la etiqueta HTML.
    for r in refs.values():
        for campo in ("title", "venue"):
            if r.get(campo):
                s = re.sub(r"<[^>]+>", "", str(r[campo]))
                r[campo] = re.sub(r"\s+", " ", s).strip()

    orden, faltantes = [], []
    for m in re.finditer(r"\[\[([^\]]+)\]\]", texto):
        d = m.group(1).strip().lower()
        if d not in refs:
            if d not in faltantes:
                faltantes.append(d)
            continue
        if d not in orden:
            orden.append(d)

    if faltantes:
        print("DOI citados que NO estan en refs_verified.json")
        for d in faltantes:
            print("   ", d)
        print("No se genera el manuscrito. Verificar contra Crossref primero.")
        return 1

    num = {d: i + 1 for i, d in enumerate(orden)}

    def reemplazo(m):
        d = m.group(1).strip().lower()
        n = num[d]
        ancla = f"ref{n}"
        return f"[[{n}](#{ancla})]" if a.estilo == "ieee" else \
               f"({apellido(refs[d]['authors'][0]) if refs[d]['authors'] else 'Anon'}, {refs[d]['year']})"

    cuerpo = re.sub(r"\[\[([^\]]+)\]\]", reemplazo, texto)

    if a.estilo == "ieee":
        # Ancla NATIVA de pandoc, no HTML crudo. Con <a id=...> el escritor de docx
        # no puede emitir el HTML y acaba inventando un nombre de marcador cifrado,
        # de modo que los hipervinculos del cuerpo apuntan a un ancla inexistente y
        # en Word no saltan a ninguna parte. La forma []{#ref1} produce un marcador
        # llamado exactamente ref1.
        lineas = [f'[]{{#ref{num[d]}}}{ieee(refs[d], num[d])}' for d in orden]
        cabecera = "Referencias numeradas por orden de aparicion en el texto."
    else:
        alfab = sorted(orden, key=lambda d: (apellido(refs[d]["authors"][0]) if refs[d]["authors"] else "",
                                             refs[d]["year"]))
        lineas = [apa(refs[d]) for d in alfab]
        cabecera = "Referencias en orden alfabetico."

    marca = "[PENDIENTE DE GENERACIÓN AUTOMÁTICA."
    if marca in cuerpo:
        cuerpo = re.sub(r"\[PENDIENTE DE GENERACIÓN AUTOMÁTICA\.[^\]]*\]",
                        cabecera + "\n\n" + "\n\n".join(lineas), cuerpo, flags=re.S)
    else:
        cuerpo += "\n\n" + cabecera + "\n\n" + "\n\n".join(lineas)

    out = src.with_name(src.stem + f"_{a.estilo}.md")
    out.write_text(cuerpo, encoding="utf-8")
    print(f"Manuscrito con referencias -> {out.name}")
    print(f"  citas unicas {len(orden)}  |  estilo {a.estilo}")
    recientes = sum(1 for d in orden if refs[d]["year"] and refs[d]["year"] >= 2022)
    print(f"  citadas de 2022 o posterior {recientes} ({100*recientes/max(len(orden),1):.0f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
