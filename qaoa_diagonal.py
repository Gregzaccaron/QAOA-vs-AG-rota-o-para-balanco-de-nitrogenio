"""Evolucao do estado do QAOA aplicando a camada de custo na forma diagonal.

VEREDITO MEDIDO (04/11/2026): esta rota e EXATA mas NAO e mais rapida.
=====================================================================
Fidelidade 1,000000000000 contra o circuito no Aer em todas as instancias de 8
a 24 qubits, o que confirma a equivalencia matematica. Mas o tempo, em 24
qubits com p = 2:

    circuito de portas no Aer ....  3,7 s
    porta 'diagonal' nativa do Aer  29,4 s   (8x mais lento)
    NumPy puro (este modulo) .....  9,1 s   (2,5x mais lento)

Por que a premissa estava errada: eu supus que o Aer fizesse uma varredura da
memoria por porta, 1.333 ao todo. A medicao mostra que ele faz **fusao de
portas**, combinando varias numa so matriz aplicada em blocos que cabem no
cache. Prova: as 1.333 portas em 3,7 s implicariam 194 GB/s de banda, e a
memoria da maquina entrega 28 GB/s medidos. Nao ha 1.333 varreduras, o Aer
nunca as fez.

O que ajudou a me enganar: o ajuste t = alfa * portas * 2^n deu R2 = 0,9985. A
contagem de portas e **proporcional** ao trabalho real, porque as duas crescem
do mesmo jeito com n, mas nao e o numero de varreduras. Correlacao, nao
mecanismo.

Onde o tempo vai, neste modulo, em 24 qubits (medido):
    camada de custo (exp + multiplicacao)  0,76 s   <- aqui a ideia ganha
    camada de mistura (24 rotacoes) ...... 7,66 s   <- aqui perde tudo

A mistura e 84% do tempo. O NumPy faz operacoes elemento a elemento em
complexos em uma unica thread, a 28 GB/s, que e perto do teto da memoria; o
Aer fusiona as rotacoes e usa varias threads. Nao da para vencer isso sem
codigo C multithread, e esse nao e o escopo de um TCC.

O modulo fica no repositorio por dois motivos: :func:`validar_contra_aer`
documenta a equivalencia (util em Materiais e Metodos, pois mostra que o
resultado nao depende da rota de simulacao), e o veredito acima e um resultado
negativo que vale registrar para nao ser tentado de novo.

O raciocinio original, que continua correto como matematica
-----------------------------------------------------------
O Hamiltoniano de custo do QAOA, vindo de um QUBO, contem apenas operadores Z.
Como Z e diagonal na base computacional (Z|0> = +|0>, Z|1> = -|1>), toda soma
de produtos de Z tambem e diagonal:

    C |i> = E_i |i>

onde E_i e o custo da solucao i. Logo a exponencial do operador se reduz a uma
exponencial numero por numero:

    exp(-i g C) = diag( exp(-i g E_0), ..., exp(-i g E_{2^n - 1}) )

e aplicar a camada de custo inteira e **uma** multiplicacao elemento a
elemento do vetor de estado.

O circuito transpilado faz o mesmo calculo quebrado nas suas centenas de
parcelas: cada acoplamento J_ab vira CX-RZ-CX, e em 28 qubits o circuito tem
1688 portas. Cada porta e uma varredura completa das 2^n amplitudes. Portanto
o simulador faz 1688 varreduras para obter algo igual a 1 multiplicacao
elemento a elemento mais n rotacoes de mistura.

    por avaliacao, em 28 qubits com p = 2
        via circuito : 1688 varreduras
        via diagonal : p * (1 + n) = 58 varreduras

A camada de mistura nao pode receber o mesmo tratamento: B = soma de X_a nao e
diagonal, porque X troca |0> por |1>. O que se pode fazer e fatora-la, ja que
os X_a atuam em qubits distintos e portanto comutam:

    exp(-i b * sum_a X_a) = produto_a exp(-i b X_a)

sao n rotacoes de um qubit, uma varredura cada, e esse custo e irredutivel
nesta representacao. Depois da otimizacao a mistura passa a ser o custo
dominante, com n das n+1 varreduras por camada.

Equivalencia, nao aproximacao
-----------------------------
Nada aqui e truncado nem aproximado: e o mesmo operador unitario calculado de
outra forma. A funcao :func:`validar_contra_aer` comprova isso medindo a
fidelidade |<psi_aer|psi_diagonal>|^2 contra o simulador do circuito, que deve
dar 1 a menos do erro de ponto flutuante.

A ressalva honesta, que pertence a Materiais e Metodos: este atalho so existe
porque mantemos a tabela das 2^n energias na memoria, que e justamente o
recurso que torna a simulacao classica exponencial. Um computador quantico
real nao dispoe dessa tabela, e por isso nao poderia usar o atalho. A
otimizacao acelera a **simulacao**, nao o algoritmo.
"""

from __future__ import annotations

import time

import numpy as np

from qubo_rotacao import ModeloQUBO


# ---------------------------------------------------------------------------
# evolucao
# ---------------------------------------------------------------------------

def _aplicar_mistura(psi: np.ndarray, n: int, q: int, beta: float,
                     buffer: np.ndarray) -> None:
    """Aplica exp(-i beta X_q) ao vetor de estado, no lugar.

    exp(-i b X) = cos(b) I - i sen(b) X, isto e a matriz

        [ cos b      -i sen b ]
        [ -i sen b    cos b   ]

    Para agir sobre o qubit q basta ver o vetor como tres eixos
    (alto, bit_q, baixo), com baixo de tamanho 2^q, e combinar as duas fatias
    do eixo do meio. Nenhuma matriz 2^n x 2^n e construida.

    O ``buffer`` evita alocar metade do vetor a cada rotacao: em 28 qubits
    seriam 2 GB de lixo por porta, 56 vezes por avaliacao.
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)

    v = psi.reshape(-1, 2, 1 << q)
    a = v[:, 0, :]
    b = v[:, 1, :]
    tmp = buffer[: a.size].reshape(a.shape)

    # tmp = c*a + s*b ; depois b = s*a + c*b ; por fim a = tmp
    np.multiply(a, c, out=tmp)
    tmp += s * b
    np.multiply(b, c, out=b)
    b += s * a
    a[...] = tmp


def evoluir(
    energias_circuito: np.ndarray,
    n: int,
    gammas: np.ndarray,
    betas: np.ndarray,
    buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Devolve o vetor de estado do QAOA para os parametros dados.

    Parameters
    ----------
    energias_circuito:
        Diagonal do Hamiltoniano de custo **na mesma convencao do circuito**,
        ou seja as energias de Ising sem o termo constante. Ver
        :func:`energias_do_circuito`.
    n:
        Numero de qubits.
    gammas, betas:
        Parametros das p camadas, em ordem.
    buffer:
        Area de trabalho de 2^(n-1) complexos, reaproveitada entre chamadas.
    """
    dim = 1 << n
    if buffer is None:
        buffer = np.empty(dim // 2, dtype=np.complex128)

    # estado inicial: superposicao uniforme, o H em todos os qubits
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)

    for gamma, beta in zip(gammas, betas):
        # camada de custo: UMA multiplicacao elemento a elemento
        psi *= np.exp(-1j * gamma * energias_circuito)
        # camada de mistura: n rotacoes de um qubit
        for q in range(n):
            _aplicar_mistura(psi, n, q, beta, buffer)

    return psi


def energias_do_circuito(modelo: ModeloQUBO,
                         energias_qubo: np.ndarray) -> np.ndarray:
    """Converte as energias do QUBO para a convencao que o circuito aplica.

    O ``construir_hamiltoniano`` omite deliberadamente o termo constante da
    conversao QUBO -> Ising, enquanto ``energias_qubo`` devolve o custo QUBO
    completo, que **inclui** esse termo. A diferenca e uma constante igual para
    todas as configuracoes, logo altera o estado apenas por uma fase global,
    que nao afeta probabilidade nem valor esperado. Mesmo assim descontamos a
    constante, para que a comparacao com o Aer seja exata amplitude por
    amplitude e nao apenas a menos de fase.
    """
    from qaoa_rotacao import qubo_para_ising

    _h, _J, const_ising = qubo_para_ising(modelo)
    return energias_qubo - const_ising


# ---------------------------------------------------------------------------
# validacao
# ---------------------------------------------------------------------------

def validar_contra_aer(modelo: ModeloQUBO, p: int = 2,
                       seed: int = 0) -> dict:
    """Compara o estado desta implementacao com o do circuito no Aer.

    A fidelidade |<psi_aer|psi_diag>|^2 deve dar 1. Se der, as duas rotas
    produzem o mesmo estado e portanto o mesmo algoritmo: o estado e a saida
    do QAOA.
    """
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer import AerSimulator

    from qaoa_rotacao import construir_hamiltoniano, energias_qubo

    n = modelo.n_variaveis
    energias = energias_qubo(modelo, limite_gb=8.0)
    energias_circ = energias_do_circuito(modelo, energias)

    hamiltoniano = construir_hamiltoniano(modelo)
    ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=p).decompose(reps=3)
    backend = AerSimulator(method="statevector")
    circuito = ansatz.copy()
    circuito.save_statevector()
    circuito = transpile(circuito, backend)

    rng = np.random.default_rng(seed)
    parametros = rng.uniform(0, np.pi, ansatz.num_parameters)

    # O QAOAAnsatz nomeia os parametros com as letras gregas beta e gama, e a
    # ParameterView os devolve ordenados por nome, logo os betas vem antes dos
    # gamas. Separamos pelo nome em vez de confiar na posicao, e so caimos na
    # divisao posicional se os nomes nao seguirem o padrao.
    nomes = [par.name for par in ansatz.parameters]
    betas = np.array([v for nome, v in zip(nomes, parametros)
                      if nome.startswith("β")])
    gammas = np.array([v for nome, v in zip(nomes, parametros)
                       if nome.startswith("γ")])
    if betas.size != p or gammas.size != p:
        if parametros.size != 2 * p:
            raise RuntimeError(
                f"Esperava {2 * p} parametros, recebi {parametros.size}."
            )
        betas, gammas = parametros[:p], parametros[p:]

    ligacoes = [{q: [v] for q, v in zip(ansatz.parameters, parametros)}]
    t0 = time.perf_counter()
    res = backend.run(circuito, parameter_binds=ligacoes).result()
    psi_aer = np.asarray(res.get_statevector())
    t_aer = time.perf_counter() - t0

    t0 = time.perf_counter()
    psi_diag = evoluir(energias_circ, n, gammas, betas)
    t_diag = time.perf_counter() - t0

    produto = complex(np.vdot(psi_aer, psi_diag))
    fidelidade = float(abs(produto) ** 2)
    ve_aer = float((np.abs(psi_aer) ** 2) @ energias)
    ve_diag = float((np.abs(psi_diag) ** 2) @ energias)
    portas = sum(circuito.count_ops().values())

    return {
        "n_qubits": n,
        "p": p,
        "fidelidade": fidelidade,
        "erro_max_amplitude": float(np.abs(psi_aer - psi_diag).max()),
        "valor_esperado_aer": ve_aer,
        "valor_esperado_diagonal": ve_diag,
        "diferenca_valor_esperado": ve_diag - ve_aer,
        "portas_circuito": portas,
        "varreduras_diagonal": p * (1 + n),
        "t_aer_s": t_aer,
        "t_diagonal_s": t_diag,
        "ganho": t_aer / t_diag if t_diag > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# uso direto
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from parametros import CENARIOS_POR_TAMANHO

    ap = argparse.ArgumentParser(
        description="Valida e cronometra a camada de custo diagonal"
    )
    ap.add_argument("--max-qubits", type=int, default=24)
    ap.add_argument("--p", type=int, default=2)
    args = ap.parse_args()

    print("Validacao da camada de custo diagonal contra o circuito no Aer")
    print("A fidelidade deve dar 1,0 e o valor esperado deve coincidir.\n")

    cab = (f"{'n':>4} {'portas':>7} {'varred':>7} {'fidelidade':>14} "
           f"{'dif <C>':>12} {'t_aer':>9} {'t_diag':>9} {'ganho':>7}")
    print(cab)
    print("-" * len(cab))

    for n_cult in sorted(CENARIOS_POR_TAMANHO):
        for n_per in (2, 3, 4):
            n = n_cult * n_per
            if n > args.max_qubits or n < 8:
                continue
            if n % 4 != 0 and n_per != 3:
                continue
            modelo = ModeloQUBO(
                culturas=CENARIOS_POR_TAMANHO[n_cult], n_periodos=n_per
            )
            try:
                r = validar_contra_aer(modelo, p=args.p)
            except Exception as e:                     # noqa: BLE001
                # a mensagem pode conter letras gregas, que o console cp1252
                # do Windows nao codifica; por isso o ascii().
                print(f"{n:>4}   FALHOU: {type(e).__name__}: "
                      f"{ascii(str(e))}", flush=True)
                continue
            ok = "OK" if r["fidelidade"] > 1 - 1e-9 else "DIVERGIU"
            print(f"{r['n_qubits']:>4} {r['portas_circuito']:>7} "
                  f"{r['varreduras_diagonal']:>7} "
                  f"{r['fidelidade']:>14.12f} "
                  f"{r['diferenca_valor_esperado']:>12.2e} "
                  f"{r['t_aer_s']:>8.3f}s {r['t_diagonal_s']:>8.3f}s "
                  f"{r['ganho']:>6.1f}x  {ok}", flush=True)
