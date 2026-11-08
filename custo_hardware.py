"""Estima o custo do QAOA em hardware quântico real, sem gastar tempo de QPU.

O que este script mede, e o que ele apenas supõe
------------------------------------------------
O trabalho executou o QAOA em simulador. A pergunta natural é o que
aconteceria num processador quântico real, e ela tem duas partes com status
epistêmico diferente.

**Medido aqui.** A transpilação. Um circuito só roda num processador real
depois de ser reescrito para o conjunto de portas nativas do dispositivo e
para a sua topologia de acoplamento, que na família IBM é a *heavy-hex*, de
grau 2 a 3. Como o nosso Hamiltoniano de custo acopla muitos pares dentro de
cada bloco e o *mixer* XY exige um anel por período, a reescrita insere portas
SWAP e **infla a profundidade**. Quanto ela infla não é palpite: este script
transpila os circuitos reais do trabalho para uma *heavy-hex* de verdade, com
o conjunto de portas nativo, e **conta** a profundidade e as portas de dois
qubits resultantes.

**Suposto, e declarado como tal.** Os parâmetros do dispositivo: duração e
erro da porta de dois qubits. São valores de literatura, informados na
constante ``DISPOSITIVOS`` abaixo, com a fonte. Não foram medidos aqui, e
variam de 3 a 5 vezes entre o melhor e o pior qubit de um mesmo chip.

Portanto: a inflação da transpilação, que era a maior incerteza da estimativa,
passa a ser um número contado. O restante continua sendo conta de ordem de
grandeza, e deve ser apresentado como tal.

Uso
---
    python custo_hardware.py
    python custo_hardware.py --dispositivo heron --qubits 16 20 24
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import CouplingMap

from parametros import CENARIOS_POR_TAMANHO
from qaoa_rotacao import (construir_hamiltoniano, estado_inicial_onehot,
                          mixer_xy)
from qubo_rotacao import ModeloQUBO

# Parâmetros de dispositivo. NÃO são medidos por este script: vêm da
# documentação pública da IBM e devem ser citados como tais. A duração é a da
# porta de dois qubits nativa; o erro é a mediana informada para o chip.
DISPOSITIVOS = {
    "eagle": {
        "nome": "IBM Eagle r3 (127 qubits)",
        "basis": ["ecr", "rz", "sx", "x"],
        "t_2q_ns": 533.0,
        "err_2q": 7.0e-3,
        "fonte": "IBM Quantum, docs de calibração da família Eagle",
    },
    "heron": {
        "nome": "IBM Heron r2 (156 qubits)",
        "basis": ["cz", "rz", "sx", "x"],
        "t_2q_ns": 68.0,
        "err_2q": 3.0e-3,
        "fonte": "IBM Quantum, docs de calibração da família Heron",
    },
}

#: Medições por avaliação e avaliações por execução, como no experimento.
SHOTS = 4096
AVALIACOES = 131          # média medida: entre 118 e 144
#: Sobrecusto por medição além da duração do circuito: leitura e reinício.
OVERHEAD_SHOT_US = 4.0


def heavy_hex_para(n_qubits: int) -> CouplingMap:
    """Menor malha heavy-hex que acomoda n_qubits."""
    for d in range(3, 15, 2):
        cm = CouplingMap.from_heavy_hex(d)
        if cm.size() >= n_qubits:
            return cm
    raise ValueError(f"nenhuma heavy-hex acomoda {n_qubits} qubits")


def circuito_qaoa(n_culturas: int, mixer: str, p: int = 2):
    """Reconstrói o circuito do experimento, sem executá-lo."""
    modelo = ModeloQUBO(culturas=CENARIOS_POR_TAMANHO[n_culturas])
    ham = construir_hamiltoniano(modelo)
    if mixer == "XY":
        ansatz = QAOAAnsatz(cost_operator=ham, reps=p,
                            mixer_operator=mixer_xy(modelo),
                            initial_state=estado_inicial_onehot(modelo))
    else:
        ansatz = QAOAAnsatz(cost_operator=ham, reps=p)
    ansatz = ansatz.decompose(reps=4)
    # Parâmetros concretos: a contagem de portas não depende dos valores, mas
    # a transpilação exige um circuito ligado.
    rng = np.random.default_rng(0)
    return ansatz.assign_parameters(rng.random(ansatz.num_parameters)), modelo


def analisar(n_culturas: int, mixer: str, disp: dict, p: int = 2) -> dict:
    qc, modelo = circuito_qaoa(n_culturas, mixer, p)
    n = modelo.n_variaveis
    cm = heavy_hex_para(n)

    antes_d = qc.depth()
    antes_2q = sum(1 for inst in qc.data if len(inst.qubits) == 2)

    tq = transpile(qc, coupling_map=cm, basis_gates=disp["basis"],
                   optimization_level=3, seed_transpiler=42)
    dep = tq.depth()
    g2q = sum(1 for inst in tq.data if len(inst.qubits) == 2)

    dur_us = dep * disp["t_2q_ns"] / 1000.0
    por_shot_us = dur_us + OVERHEAD_SHOT_US
    qpu_s = por_shot_us * SHOTS * AVALIACOES / 1e6
    fid = math.exp(-g2q * disp["err_2q"]) if g2q else 1.0

    return dict(n=n, mixer=mixer, malha=cm.size(), antes_d=antes_d,
                antes_2q=antes_2q, dep=dep, g2q=g2q,
                inflacao_d=dep / antes_d if antes_d else 0.0,
                inflacao_2q=g2q / antes_2q if antes_2q else 0.0,
                dur_us=dur_us, qpu_s=qpu_s, fid=fid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dispositivo", choices=sorted(DISPOSITIVOS),
                    default="heron")
    ap.add_argument("--qubits", nargs="+", type=int, default=[16, 20, 24],
                    help="Tamanhos em variáveis binárias (múltiplos de 4).")
    ap.add_argument("--p", type=int, default=2)
    args = ap.parse_args()

    disp = DISPOSITIVOS[args.dispositivo]
    print(f"Dispositivo suposto: {disp['nome']}")
    print(f"  porta de 2 qubits: {disp['t_2q_ns']:.0f} ns, "
          f"erro {disp['err_2q']:.1e}   (fonte: {disp['fonte']})")
    print(f"  orçamento por execução: {AVALIACOES} avaliações x "
          f"{SHOTS} medições\n")

    print(f"{'n':>3} {'mixer':>5} | {'prof.':>6} {'2q':>6} -> "
          f"{'prof.':>7} {'2q':>7} | {'infl.':>6} | "
          f"{'circ.':>9} {'QPU':>9} | {'P(sem erro)':>12}")
    print("-" * 96)
    linhas = []
    for n in args.qubits:
        nc = n // 4
        if nc not in CENARIOS_POR_TAMANHO:
            print(f"  (sem cenário para {n} variáveis)")
            continue
        for mixer in ("X", "XY"):
            r = analisar(nc, mixer, disp, args.p)
            linhas.append(r)
            print(f"{r['n']:>3} {r['mixer']:>5} | {r['antes_d']:>6} "
                  f"{r['antes_2q']:>6} -> {r['dep']:>7} {r['g2q']:>7} | "
                  f"{r['inflacao_d']:>5.1f}x | {r['dur_us']:>7.0f} us "
                  f"{r['qpu_s']:>7.1f} s | {r['fid']:>12.2e}")

    print("\nLeitura")
    print("-------")
    print("A coluna 'infl.' é a razão entre a profundidade transpilada para a")
    print("heavy-hex e a profundidade no circuito abstrato. É o número que")
    print("este script mede, e era a maior incerteza da estimativa.")
    print()
    print("'QPU' é o tempo de processador quântico por execução do laço")
    print("híbrido completo, sem fila nem ida e volta clássica. Compare com os")
    print("557,2 s de simulação da instância de 24 variáveis com mixer XY.")
    print()
    print("'P(sem erro)' é exp(-portas_2q x erro_2q): a probabilidade de o")
    print("circuito inteiro rodar sem nenhum erro de porta de dois qubits.")
    print("Confronte com a probabilidade ideal de amostrar o ótimo, que o")
    print("trabalho mediu em 2,4e-3 para 24 qubits com mixer XY. Se")
    print("'P(sem erro)' for da mesma ordem ou menor, o sinal se perde no")
    print("ruído e o ótimo não é observável com este orçamento de medições.")


if __name__ == "__main__":
    main()
