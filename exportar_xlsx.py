"""Converte os resultados para .xlsx, que o Excel abre sem interpretar nada.

Por que não basta o CSV
-----------------------
``resultados/comparativo.csv`` está no formato canônico: vírgula separa as
colunas e ponto separa a parte decimal. É o que o Python, o pandas e o R
esperam. O Excel em português do Brasil assume o contrário e, ao abrir o
arquivo, ou joga tudo numa coluna só ou interpreta ``384.68`` como texto e
``0,009974`` como data. O Excel da web é ainda menos tolerante.

Um ``.xlsx`` resolve de vez, porque cada célula carrega o seu próprio tipo:
número é número, texto é texto, e não há separador para o programa adivinhar.

Não há dependência externa. O formato ``.xlsx`` é um arquivo ZIP com algumas
partes em XML, e é isso que este script escreve, usando apenas ``zipfile`` e
``csv`` da biblioteca padrão. Não é preciso instalar ``openpyxl``.

Saída
-----
``resultados/comparativo.xlsx``, com duas planilhas:

``execucoes``
    As 480 execuções, uma por linha, com as colunas do arquivo canônico.

``resumo``
    Uma linha por (instância, método): o ótimo conhecido, o desvio percentual
    médio e o seu desvio-padrão, quantas execuções acertaram o ótimo, quantas
    produziram alocação válida e o tempo médio. É a tabela que se lê para
    comparar os métodos, e corresponde à Tabela 4 do artigo.

Uso
---
    python exportar_xlsx.py
    python exportar_xlsx.py --entrada resultados/comparativo.csv \\
                            --saida resultados/comparativo.xlsx
"""

from __future__ import annotations

import argparse
import csv
import statistics
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# Colunas numéricas: escritas como número, não como texto. As demais viram
# texto, inclusive "21/30", que o Excel converteria em data.
NUMERICAS = {
    "n_culturas", "n_talhoes", "n_periodos", "n_variaveis", "execucao",
    "custo", "custo_otimo", "desvio_percentual", "demanda_n_kg_ha",
    "saldo_n_kg_ha", "adubacao_p2o5_kg_ha", "adubacao_k2o_kg_ha", "tempo_s",
    "n_avaliacoes", "geracoes", "geracao_convergencia", "qaoa_p",
    "valor_esperado", "profundidade_circuito", "n_portas", "prob_melhor",
    "memoria_estimada_gb",
}


def _letra(n: int) -> str:
    """1 -> A, 27 -> AA."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _celula(valor: str, col: int, lin: int, cabecalho: str, negrito: bool):
    ref = f"{_letra(col)}{lin}"
    estilo = ' s="1"' if negrito else ""
    if not negrito and cabecalho in NUMERICAS and valor not in ("", "None"):
        try:
            float(valor)
        except ValueError:
            pass
        else:
            return f'<c r="{ref}"{estilo}><v>{valor}</v></c>'
    return (f'<c r="{ref}"{estilo} t="inlineStr"><is><t>'
            f"{escape(valor)}</t></is></c>")


def _planilha(linhas: list[list[str]]) -> str:
    cab = linhas[0] if linhas else []
    corpo = []
    for i, linha in enumerate(linhas, start=1):
        cs = "".join(
            _celula(v, j, i, cab[j - 1] if j - 1 < len(cab) else "", i == 1)
            for j, v in enumerate(linha, start=1)
        )
        corpo.append(f'<row r="{i}">{cs}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(corpo)}</sheetData></worksheet>'
    )


def escrever_xlsx(destino: Path, abas: dict[str, list[list[str]]]) -> None:
    nomes = list(abas)
    sheets = "".join(
        f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, n in enumerate(nomes, start=1)
    )
    rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        f'relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(nomes) + 1)
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType='
        '"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(nomes) + 1)
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   f"{overrides}</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   "</Relationships>")
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   f"<sheets>{sheets}</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   f'{rels}<Relationship Id="rId{len(nomes) + 1}" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
                   'Target="styles.xml"/></Relationships>')
        # Um único estilo além do padrão: o 1 é negrito, usado no cabeçalho.
        z.writestr("xl/styles.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="2"><font/><font><b/></font></fonts>'
                   '<fills count="1"><fill/></fills>'
                   '<borders count="1"><border/></borders>'
                   '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
                   '<cellXfs count="2"><xf xfId="0"/>'
                   '<xf xfId="0" fontId="1" applyFont="1"/></cellXfs>'
                   "</styleSheet>")
        for i, nome in enumerate(nomes, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _planilha(abas[nome]))


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resumir(linhas: list[dict[str, str]]) -> list[list[str]]:
    """Agrega por (instância, método), como a Tabela 4 do artigo."""
    grupos: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in linhas:
        grupos.setdefault((r["n_variaveis"], r["metodo"]), []).append(r)

    saida = [["n_variaveis", "metodo", "execucoes", "custo_otimo",
              "desvio_medio_pct", "desvio_padrao_pct", "acertou_otimo",
              "alocacao_valida", "tempo_medio_s"]]
    for (n, metodo) in sorted(grupos, key=lambda k: (int(k[0]), k[1])):
        g = grupos[(n, metodo)]
        desvios = [d for d in (_num(r["desvio_percentual"]) for r in g)
                   if d is not None]
        tempos = [t for t in (_num(r["tempo_s"]) for r in g) if t is not None]
        otimos = sum(1 for r in g if r.get("atingiu_otimo", "").lower()
                     in ("true", "1", "sim"))
        validas = sum(1 for r in g if r.get("alocacao_valida", "").lower()
                      in ("true", "1", "sim"))
        saida.append([
            n, metodo, str(len(g)), g[0].get("custo_otimo", ""),
            f"{statistics.fmean(desvios):.4f}" if desvios else "",
            f"{statistics.pstdev(desvios):.4f}" if len(desvios) > 1 else "0",
            f"{otimos}/{len(g)}", f"{validas}/{len(g)}",
            f"{statistics.fmean(tempos):.6f}" if tempos else "",
        ])
    return saida


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--entrada", type=Path,
                   default=Path("resultados/comparativo.csv"))
    p.add_argument("--saida", type=Path,
                   default=Path("resultados/comparativo.xlsx"))
    args = p.parse_args()

    if not args.entrada.exists():
        raise SystemExit(f"não encontrei {args.entrada}. "
                         "Rode experimento_comparativo.py primeiro.")

    with open(args.entrada, encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        campos = list(leitor.fieldnames or [])
        registros = list(leitor)

    execucoes = [campos] + [[r.get(c, "") or "" for c in campos]
                            for r in registros]
    escrever_xlsx(args.saida,
                  {"execucoes": execucoes, "resumo": resumir(registros)})

    print(f"  {args.saida}")
    print(f"    aba 'execucoes': {len(registros)} linhas, {len(campos)} colunas")
    print(f"    aba 'resumo': {len(resumir(registros)) - 1} linhas")


if __name__ == "__main__":
    main()
