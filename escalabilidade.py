"""Mede o custo de simular o QAOA em função do número de qubits.

Por que este experimento existe
-------------------------------
A notação assintótica diz que a simulação por vetor de estado é Theta(2^n), e
só. Mas 2^n descreve igualmente bem algo que leva um segundo e algo que leva
um século: o que decide se um cenário é executável é a **constante** que a
notação descarta. Este script mede essa constante.

O modelo de custo
-----------------
Cada porta de um qubit precisa percorrer todas as 2^n amplitudes do vetor de
estado. Logo o trabalho de uma avaliação da função de custo é

    W(n, p) = (numero de portas do circuito) x 2^n

e o tempo medido deve ser t = alfa * W, com alfa em nanossegundos por
amplitude-porta. O script conta as portas **do circuito transpilado de fato**,
sem estimá-las, e calcula alfa = t / W para cada n.

Como ler o resultado
--------------------
Se alfa for aproximadamente constante ao longo de n, o modelo de custo está
validado e alfa caracteriza a máquina. Onde alfa **sobe**, o vetor de estado
deixou de caber num nível de cache e a simulação passou a ser limitada pela
banda de memória: esse joelho é um resultado em si, e é o que inviabiliza a
extrapolação ingênua a partir de instâncias pequenas.

Uso
---
    python escalabilidade.py                  # até 24 qubits
    python escalabilidade.py --max-qubits 28  # inclui a instância de 7 culturas
    python escalabilidade.py --p 1 --p 2      # mais de um valor de p
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

from parametros import CENARIOS_POR_TAMANHO
from qubo_rotacao import ModeloQUBO

RESULTADOS = pathlib.Path(__file__).parent / "resultados"


# ---------------------------------------------------------------------------
# instancias
# ---------------------------------------------------------------------------

def escada_de_instancias(max_qubits: int,
                         min_qubits: int = 8) -> list[tuple[int, int, int]]:
    """Devolve (n_qubits, n_culturas, n_periodos) em ordem crescente de n.

    O numero de qubits e ``n_culturas * n_periodos`` (um talhao). Variar os
    dois eixos da mais pontos na curva do que a escada de culturas sozinha,
    que salta de quatro em quatro qubits.
    """
    vistos: dict[int, tuple[int, int]] = {}
    for n_cult in sorted(CENARIOS_POR_TAMANHO):
        for n_per in (2, 3, 4):
            n = n_cult * n_per
            if n > max_qubits or n < min_qubits:
                continue
            # entre duas combinacoes do mesmo n fica a de mais periodos: e a
            # que corresponde ao cenario real do trabalho.
            if n not in vistos or n_per > vistos[n][1]:
                vistos[n] = (n_cult, n_per)
    return [(n, c, p) for n, (c, p) in sorted(vistos.items())]


# ---------------------------------------------------------------------------
# medicao
# ---------------------------------------------------------------------------

def medir_instancia(
    n_culturas: int,
    n_periodos: int,
    p: int,
    n_avaliacoes: int,
    precisao: str,
) -> dict:
    """Cronometra uma instancia e devolve a linha de resultado."""
    # Importado aqui para que o script ainda rode (e explique o erro) numa
    # maquina sem Qiskit instalado.
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer import AerSimulator

    from qaoa_rotacao import construir_hamiltoniano, energias_qubo

    modelo = ModeloQUBO(
        culturas=CENARIOS_POR_TAMANHO[n_culturas], n_periodos=n_periodos
    )
    n = modelo.n_variaveis

    t0 = time.perf_counter()
    hamiltoniano = construir_hamiltoniano(modelo)
    ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=p).decompose(reps=3)
    backend = AerSimulator(method="statevector", precision=precisao)
    circuito = ansatz.copy()
    circuito.save_statevector()
    circuito = transpile(circuito, backend)
    t_montagem = time.perf_counter() - t0

    # pre-calculo da diagonal do Hamiltoniano de custo: custo unico por
    # instancia, reaproveitado por todas as avaliacoes.
    t0 = time.perf_counter()
    energias = energias_qubo(modelo, limite_gb=8.0)
    t_energias = time.perf_counter() - t0

    nomes = list(ansatz.parameters)
    rng = np.random.default_rng(0)

    def uma_avaliacao(parametros: np.ndarray) -> float:
        ligacoes = [{q: [v] for q, v in zip(nomes, parametros)}]
        res = backend.run(circuito, parameter_binds=ligacoes).result()
        psi = np.asarray(res.get_statevector())
        return float((np.abs(psi) ** 2) @ energias)

    # uma avaliacao de aquecimento, descartada: a primeira paga a compilacao
    # interna do Aer e a primeira falta de pagina da memoria recem-alocada.
    uma_avaliacao(rng.uniform(0, np.pi, ansatz.num_parameters))

    tempos: list[float] = []
    for _ in range(n_avaliacoes):
        parametros = rng.uniform(0, np.pi, ansatz.num_parameters)
        t0 = time.perf_counter()
        uma_avaliacao(parametros)
        tempos.append(time.perf_counter() - t0)

    contagem = circuito.count_ops()
    # save_statevector nao e uma porta; desconta-se para que a contagem
    # reflita so o que varre o vetor de estado.
    portas = sum(contagem.values()) - contagem.get("save_statevector", 0)

    t_mediano = statistics.median(tempos)
    trabalho = portas * (2 ** n)          # portas x amplitudes
    alfa_ns = t_mediano / trabalho * 1e9

    del energias
    return {
        "n_qubits": n,
        "n_culturas": n_culturas,
        "n_periodos": n_periodos,
        "p": p,
        "portas": portas,
        "profundidade": circuito.depth(),
        "amplitudes": 2 ** n,
        "trabalho": trabalho,
        "t_montagem_s": round(t_montagem, 4),
        "t_energias_s": round(t_energias, 4),
        "t_avaliacao_s": round(t_mediano, 6),
        "t_avaliacao_min_s": round(min(tempos), 6),
        "t_avaliacao_max_s": round(max(tempos), 6),
        "alfa_ns": round(alfa_ns, 5),
        "mem_estado_gb": round((2 ** n) * (8 if precisao == "single" else 16)
                               / 1024 ** 3, 4),
        "mem_energias_gb": round((2 ** n) * 8 / 1024 ** 3, 4),
    }


# ---------------------------------------------------------------------------
# ajuste e extrapolacao
# ---------------------------------------------------------------------------

def ajustar_exponencial(linhas: list[dict]) -> dict:
    """Ajusta log2(t) = a*n + b e devolve a base empirica 2^a e o R2.

    Uma base proxima de 2 confirma que o tempo dobra a cada qubit. O ajuste e
    feito no logaritmo porque e la que o crescimento exponencial e uma reta, e
    porque assim os pontos pequenos nao sao esmagados pelos grandes.
    """
    n = np.array([l["n_qubits"] for l in linhas], dtype=float)
    t = np.array([l["t_avaliacao_s"] for l in linhas], dtype=float)
    y = np.log2(t)
    a, b = np.polyfit(n, y, 1)
    previsto = a * n + b
    ss_res = float(((y - previsto) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "base_por_qubit": float(2 ** a),
        "coef_a": float(a),
        "coef_b": float(b),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def ajustar_afim(linhas: list[dict]) -> dict:
    """Ajusta t = c0 + alfa * W, separando custo fixo de custo assintotico.

    Por que nao basta t = alfa * W: em instancias pequenas o tempo e dominado
    por despesas que nao dependem de n (submissao do job ao Aer, ligacao dos
    parametros, serializacao do vetor de estado de volta para o Python). Esse
    piso faz alfa = t/W parecer enorme em n pequeno e despencar conforme n
    cresce, o que se confunde facilmente com um ganho de eficiencia. O ajuste
    afim separa as duas parcelas: ``c0`` e o piso por chamada, em segundos, e
    ``alfa`` e o custo marginal por amplitude-porta, que e o que de fato
    governa a extrapolacao.
    """
    W = np.array([l["trabalho"] for l in linhas], dtype=float)
    t = np.array([l["t_avaliacao_s"] for l in linhas], dtype=float)
    A = np.column_stack([np.ones_like(W), W])
    (c0, alfa), *_ = np.linalg.lstsq(A, t, rcond=None)
    previsto = c0 + alfa * W
    ss_res = float(((t - previsto) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "c0_s": float(c0),
        "alfa_ns": float(alfa * 1e9),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_pontos": len(linhas),
        "n_min": int(min(l["n_qubits"] for l in linhas)),
        "n_max": int(max(l["n_qubits"] for l in linhas)),
    }


def contar_portas(n_culturas: int, n_periodos: int, p: int) -> int | None:
    """Conta as portas do circuito de uma instancia SEM simula-la.

    Montar o circuito custa memoria proporcional ao numero de portas, nao a
    2^n: da para contar exatamente as portas de 32 qubits numa maquina que
    jamais conseguiria simular 32 qubits. E isso que torna a extrapolacao
    defensavel, em vez de depender de um 1,5 n^2 estimado.
    """
    try:
        from qiskit.circuit.library import QAOAAnsatz

        from qaoa_rotacao import construir_hamiltoniano
    except ImportError:
        return None
    if n_culturas not in CENARIOS_POR_TAMANHO:
        return None
    modelo = ModeloQUBO(
        culturas=CENARIOS_POR_TAMANHO[n_culturas], n_periodos=n_periodos
    )
    ansatz = QAOAAnsatz(
        cost_operator=construir_hamiltoniano(modelo), reps=p
    ).decompose(reps=3)
    return sum(ansatz.count_ops().values())


def extrapolar(alvos: list[tuple[int, int, int]], ajuste: dict, p: int,
               ram_gb: float) -> list[dict]:
    """Projeta tempo e memoria para instancias que nao foram executadas.

    Usa o modelo afim t = c0 + alfa * portas * 2^n, com as portas contadas no
    circuito de verdade sempre que a instancia existe na escada.
    """
    alfa = ajuste["alfa_ns"] / 1e9
    c0 = ajuste["c0_s"]
    saida = []
    for n, n_cult, n_per in alvos:
        portas = contar_portas(n_cult, n_per, p)
        exato = portas is not None
        if portas is None:
            portas = int(1.5 * n * n)      # recurso de ultimo caso
        t = c0 + alfa * portas * (2 ** n)
        mem_estado = (2 ** n) * 16 / 1024 ** 3
        mem_energias = (2 ** n) * 8 / 1024 ** 3
        saida.append({
            "n_qubits": n,
            "portas": portas,
            "portas_exatas": exato,
            "t_avaliacao_s": t,
            "t_avaliacao_legivel": legivel(t),
            "mem_total_gb": mem_estado + mem_energias,
            "cabe_na_ram": (mem_estado + mem_energias) < ram_gb,
        })
    return saida


def legivel(segundos: float) -> str:
    if segundos < 1:
        return f"{segundos * 1000:.0f} ms"
    if segundos < 60:
        return f"{segundos:.1f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    if segundos < 86400:
        return f"{segundos / 3600:.1f} h"
    if segundos < 86400 * 365:
        return f"{segundos / 86400:.1f} dias"
    return f"{segundos / 86400 / 365:.1f} anos"


# ---------------------------------------------------------------------------
# principal
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Escalabilidade da simulacao")
    ap.add_argument("--max-qubits", type=int, default=24)
    ap.add_argument("--min-qubits", type=int, default=8)
    ap.add_argument("--anexar", action="store_true",
                    help="soma as medicoes ao CSV existente em vez de "
                         "sobrescreve-lo")
    ap.add_argument("--p", type=int, action="append", default=None,
                    help="camadas do QAOA; pode repetir. Padrao: 2")
    ap.add_argument("--avaliacoes", type=int, default=5,
                    help="avaliacoes cronometradas por instancia (mediana)")
    ap.add_argument("--precisao", choices=("double", "single"),
                    default="double")
    ap.add_argument("--ram-gb", type=float, default=31.9)
    args = ap.parse_args()
    valores_p = args.p or [2]

    instancias = escada_de_instancias(args.max_qubits, args.min_qubits)
    print(f"Maquina : {platform.processor()}")
    print(f"Python  : {sys.version.split()[0]}   NumPy {np.__version__}")
    print(f"Precisao: {args.precisao}   avaliacoes por ponto: {args.avaliacoes}")
    print(f"\n{len(instancias)} instancias x {len(valores_p)} valores de p\n")

    cab = (f"{'n':>4} {'cult':>5} {'per':>4} {'p':>3} {'portas':>8} "
           f"{'prof':>6} {'t_aval':>12} {'alfa (ns)':>10} {'diag':>9}")
    print(cab)
    print("-" * len(cab))

    linhas: list[dict] = []
    for p in valores_p:
        for n, n_cult, n_per in instancias:
            try:
                r = medir_instancia(n_cult, n_per, p, args.avaliacoes,
                                    args.precisao)
            except MemoryError as e:
                print(f"{n:>4} {n_cult:>5} {n_per:>4} {p:>3}   "
                      f"SEM MEMORIA: {e}", flush=True)
                continue
            linhas.append(r)
            print(f"{r['n_qubits']:>4} {r['n_culturas']:>5} "
                  f"{r['n_periodos']:>4} {r['p']:>3} {r['portas']:>8} "
                  f"{r['profundidade']:>6} "
                  f"{legivel(r['t_avaliacao_s']):>12} "
                  f"{r['alfa_ns']:>10.4f} "
                  f"{legivel(r['t_energias_s']):>9}", flush=True)

    if not linhas:
        print("\nNenhuma instancia medida.")
        return 1

    RESULTADOS.mkdir(exist_ok=True)
    csv_saida = RESULTADOS / "escalabilidade.csv"
    anteriores: list[dict] = []
    if args.anexar and csv_saida.exists():
        with csv_saida.open(encoding="utf-8") as fh:
            for linha in csv.DictReader(fh):
                for k, v in linha.items():
                    try:
                        linha[k] = int(v) if "." not in v else float(v)
                    except ValueError:
                        pass
                anteriores.append(linha)
        # uma nova medicao do mesmo (n, p) substitui a antiga
        novos = {(l["n_qubits"], l["p"]) for l in linhas}
        anteriores = [l for l in anteriores
                      if (l["n_qubits"], l["p"]) not in novos]

    todas = sorted(anteriores + linhas,
                   key=lambda l: (l["p"], l["n_qubits"]))
    with csv_saida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(todas)

    # o ajuste so faz sentido com p fixo; usa-se o maior p medido.
    p_ajuste = max(valores_p)
    do_p = [l for l in todas if l["p"] == p_ajuste]
    exp = ajustar_exponencial(do_p) if len(do_p) >= 3 else None
    afim = None

    if exp:
        print(f"\nAjuste log2(t) = a*n + b  (p = {p_ajuste}, "
              f"{len(do_p)} pontos de {do_p[0]['n_qubits']} a "
              f"{do_p[-1]['n_qubits']} qubits)")
        print(f"  fator por qubit adicional : {exp['base_por_qubit']:.3f}x")
        print(f"  R^2                       : {exp['r2']:.4f}")
        a_min = min(l["alfa_ns"] for l in do_p)
        a_max = max(l["alfa_ns"] for l in do_p)
        print(f"  alfa entre {a_min:.4f} e {a_max:.4f} ns "
              f"(variacao de {a_max / a_min:.0f}x)")

        # O ajuste afim usa so a metade superior dos pontos: abaixo do joelho
        # do cache o regime e outro, e incluir esses pontos contamina alfa.
        superiores = do_p[len(do_p) // 2:]
        if len(superiores) >= 3:
            afim = ajustar_afim(superiores)
            print(f"\nAjuste afim t = c0 + alfa*W  ({afim['n_pontos']} pontos, "
                  f"{afim['n_min']} a {afim['n_max']} qubits)")
            print(f"  c0   (piso por chamada) : {afim['c0_s'] * 1000:.1f} ms")
            print(f"  alfa (marginal)         : {afim['alfa_ns']:.5f} "
                  f"ns por amplitude-porta")
            print(f"  R^2                     : {afim['r2']:.4f}")

            alvos = [(n, c, pr) for n, c, pr
                     in escada_de_instancias(32, 8)
                     if n > do_p[-1]["n_qubits"] and n % 4 == 0]
            if alvos:
                print(f"\nExtrapolacao (portas contadas no circuito real)")
                print(f"{'n':>4} {'portas':>8} {'t por avaliacao':>18} "
                      f"{'memoria':>12} {'cabe?':>7}")
                print("-" * 55)
                for e in extrapolar(alvos, afim, p_ajuste, args.ram_gb):
                    marca = "" if e["portas_exatas"] else "~"
                    print(f"{e['n_qubits']:>4} {marca}{e['portas']:>7} "
                          f"{e['t_avaliacao_legivel']:>18} "
                          f"{e['mem_total_gb']:>9.1f} GB "
                          f"{'sim' if e['cabe_na_ram'] else 'NAO':>7}")

    json_saida = RESULTADOS / "escalabilidade.json"
    json_saida.write_text(json.dumps({
        "maquina": {
            "processador": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "ram_gb": args.ram_gb,
        },
        "precisao": args.precisao,
        "avaliacoes_por_ponto": args.avaliacoes,
        "medicoes": todas,
        "ajuste_exponencial": exp,
        "ajuste_afim": afim,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGravado em {csv_saida.name} e {json_saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
