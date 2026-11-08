"""
qaoa_rotacao.py
===============

QAOA (*Quantum Approximate Optimization Algorithm*) aplicado ao QUBO de rotação
de culturas, executado em simulador quântico via Qiskit.

Otimiza exatamente a mesma matriz Q usada pelo algoritmo genético
(:mod:`algoritmo_genetico`), o que é a condição para que a comparação entre os
paradigmas seja atribuível à estratégia de busca, e não ao problema.

Etapas
------
1. **Conversão QUBO → Ising.** O QAOA opera sobre um Hamiltoniano de spins
   (operadores de Pauli Z), não sobre variáveis binárias. A substituição
   ``x = (1 − z)/2`` faz essa ponte e é implementada em
   :func:`qubo_para_ising`, em NumPy puro — sem dependência do Qiskit, o que
   permite validar a conversão isoladamente (ver :func:`verificar_conversao`).
2. **Construção do circuito.** O ``QAOAAnsatz`` alterna ``p`` camadas do
   operador de custo (parametrizado por γ) e do operador de mistura
   (parametrizado por β).
3. **Otimização híbrida.** Um otimizador clássico (COBYLA, via SciPy) ajusta
   (γ, β) para minimizar o valor esperado do Hamiltoniano — o laço
   clássico-quântico que caracteriza o algoritmo.
4. **Amostragem e decodificação.** Mede-se o estado final; entre as
   configurações amostradas, retorna-se a de menor custo segundo o QUBO.

Requisitos
----------
Este módulo exige ``qiskit`` e ``qiskit-aer`` instalados::

    python -m pip install qiskit qiskit-aer

A conversão para Ising e a estimativa de memória funcionam sem eles.

Simulador
---------
A simulação usa o **Qiskit Aer** (``AerSimulator``, método ``statevector``),
implementado em C++ e paralelizado. Foi escolhido em lugar das primitives de
referência do Qiskit (``StatevectorEstimator``/``StatevectorSampler``), que são
implementadas em Python e servem para validação, não para desempenho. O Aer
também aceita precisão simples, que corta a memória pela metade.

Limite de memória do simulador
------------------------------
A simulação por vetor de estado guarda 2^n amplitudes complexas: o consumo de
memória **dobra a cada qubit acrescentado**. Consulte
:func:`estimar_memoria_gb` antes de executar uma instância grande.

Com precisão dupla (padrão) são 16 bytes por amplitude; com precisão simples,
8 bytes. O cenário completo da escada, de 32 qubits, exige 64 GB em precisão
dupla e 32 GB em simples — fora do alcance de uma máquina pessoal em ambos os
casos. O degrau de 28 qubits exige 4 GB (2 GB em simples) e roda com folga.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from qubo_rotacao import ModeloQUBO

# O Qiskit é opcional: o módulo continua importável (e parcialmente testável)
# sem ele, o que evita que a ausência da biblioteca quebre os demais scripts.
try:  # pragma: no cover - depende do ambiente
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QAOAAnsatz, StatePreparation
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    QISKIT_DISPONIVEL = True
except ImportError:  # pragma: no cover
    QISKIT_DISPONIVEL = False


# ---------------------------------------------------------------------------
# Conversão QUBO -> Ising  (não requer Qiskit)
# ---------------------------------------------------------------------------

def qubo_para_ising(
    modelo: ModeloQUBO,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Converte o QUBO em um Hamiltoniano de Ising.

    Aplica ``x_a = (1 − z_a)/2`` sobre

        custo(x) = Σ_a Q[a,a]·x_a + Σ_{a<b} Q[a,b]·x_a·x_b + offset

    resultando em

        energia(z) = const + Σ_a h[a]·z_a + Σ_{a<b} J[a,b]·z_a·z_b

    Returns
    -------
    (h, J, const)
        ``h`` com forma ``(n,)``; ``J`` triangular superior com forma
        ``(n, n)``; ``const`` escalar.
    """
    Q = modelo.Q
    n = modelo.n_variaveis

    diagonal = np.diag(Q).copy()
    # Q é triangular superior; a versão simétrica facilita somar sobre b != a.
    fora_diagonal = Q - np.diag(diagonal)
    simetrica = fora_diagonal + fora_diagonal.T

    const = modelo.offset + diagonal.sum() / 2.0 + fora_diagonal.sum() / 4.0
    h = -diagonal / 2.0 - simetrica.sum(axis=1) / 4.0
    J = fora_diagonal / 4.0

    return h, J, float(const)


def energia_ising(
    h: np.ndarray, J: np.ndarray, const: float, x: np.ndarray
) -> float:
    """Energia de Ising correspondente a uma configuração binária ``x``."""
    z = 1.0 - 2.0 * np.asarray(x, dtype=float)
    return float(const + h @ z + z @ J @ z)


def verificar_conversao(
    modelo: ModeloQUBO, n_amostras: int = 500, seed: int | None = 0
) -> float:
    """Confere que a energia de Ising reproduz o custo do QUBO.

    Um erro de sinal ou de fator 1/2 na conversão produz um QAOA que otimiza um
    problema silenciosamente diferente do que o AG resolve, invalidando toda a
    comparação. Esta verificação deve ser executada sempre que os termos da
    função objetivo forem alterados.

    Returns
    -------
    float
        Maior discrepância absoluta observada. Deve ser da ordem do erro de
        ponto flutuante (< 1e-6).
    """
    rng = np.random.default_rng(seed)
    h, J, const = qubo_para_ising(modelo)

    X = rng.integers(0, 2, size=(n_amostras, modelo.n_variaveis))
    custos_qubo = modelo.avaliar_populacao(X)
    energias = np.array([energia_ising(h, J, const, x) for x in X])

    return float(np.max(np.abs(custos_qubo - energias)))


# ---------------------------------------------------------------------------
# Recursos de simulação
# ---------------------------------------------------------------------------

def estimar_memoria_gb(n_qubits: int, precisao: str = "double") -> float:
    """Memória exigida pela simulação por vetor de estado, em GB.

    São 2^n amplitudes complexas: 16 bytes com ``precisao="double"`` (dois
    ``float64``) ou 8 bytes com ``precisao="single"``. O resultado não inclui a
    memória de trabalho do simulador, que costuma acrescentar de 50% a 100%
    sobre esse valor.
    """
    bytes_por_amplitude = 8 if precisao == "single" else 16
    return (2**n_qubits) * bytes_por_amplitude / 1024**3


def energias_qubo(modelo: ModeloQUBO, limite_gb: float = 4.0) -> np.ndarray:
    """Custo QUBO de **todas** as 2^n configurações, na indexação do Qiskit.

    ``energias[i]`` é o custo da configuração cujo bit ``a`` (little-endian,
    isto é ``(i >> a) & 1``) vale ``x_a``. É a mesma convenção usada por
    :func:`_bitstring_para_vetor` e por :func:`construir_hamiltoniano`.

    Para que serve
    --------------
    O Hamiltoniano de custo do QAOA é **diagonal** na base computacional (só
    tem termos Z). Logo, dado o vetor de estado ψ,

        <custo> = Σ_i |ψ_i|² · energias[i]

    é um único produto escalar. Isso permite, além de calcular o valor
    esperado, obter de graça a **distribuição de probabilidade exata** sobre as
    soluções: a probabilidade de amostrar o ótimo deixa de ser estimada a
    partir de alguns milhares de shots e passa a ser um número exato.

    Atenção ao custo de memória
    ---------------------------
    São 8 bytes por configuração: 0,13 GB em 24 qubits, 2,1 GB em 28 e 34 GB
    em 32. O pré-cálculo é feito **uma vez por instância** e reaproveitado em
    todas as iterações do otimizador, em todos os reinícios e em todos os
    valores de p.
    """
    n = modelo.n_variaveis
    gb = (1 << n) * 8 / 1024**3
    if gb > limite_gb:
        raise MemoryError(
            f"O vetor de energias de {n} qubits ocuparia {gb:.1f} GB, acima do "
            f"limite de {limite_gb:.1f} GB. Use metodo_valor_esperado='aer', "
            "que não precisa dele."
        )

    energias = np.empty(1 << n, dtype=np.float64)
    lote = 1 << 16
    pesos_bits = (1 << np.arange(n, dtype=np.uint64)).astype(np.uint64)
    for inicio in range(0, 1 << n, lote):
        fim = min(inicio + lote, 1 << n)
        indices = np.arange(inicio, fim, dtype=np.uint64)[:, None]
        X = ((indices & pesos_bits) > 0).astype(np.float64)
        energias[inicio:fim] = (
            np.einsum("ij,ij->i", X @ modelo.Q, X) + modelo.offset
        )
    return energias


def tabela_de_viabilidade(max_qubits: int = 34) -> str:
    """Tabela de consumo de memória por número de qubits."""
    linhas = [
        f"{'Qubits':>7} {'Culturas':>9} {'dupla (GB)':>12} {'simples (GB)':>14}",
        "-" * 46,
    ]
    for n in range(8, max_qubits + 1, 2):
        # A escada mantém 4 períodos, logo n qubits = n/4 culturas.
        culturas = f"{n // 4}" if n % 4 == 0 else "-"
        linhas.append(
            f"{n:>7} {culturas:>9} {estimar_memoria_gb(n):>12.3f} "
            f"{estimar_memoria_gb(n, 'single'):>14.3f}"
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------

@dataclass
class ResultadoQAOA:
    """Resultado de uma execução do QAOA."""

    melhor_x: np.ndarray
    melhor_custo: float
    #: Valor esperado do Hamiltoniano com os parâmetros otimizados
    valor_esperado: float
    parametros_otimos: np.ndarray
    n_qubits: int
    #: Camadas do ansatz (o "p" do QAOA)
    p: int
    profundidade_circuito: int
    n_portas: int
    #: Iterações do otimizador clássico
    n_iteracoes: int
    #: Avaliações do valor esperado (cada uma é uma execução do circuito)
    n_avaliacoes: int
    tempo_s: float
    #: Valor esperado a cada avaliação (curva de convergência)
    historico: list[float] = field(default_factory=list)
    #: Probabilidade de amostrar a melhor solução encontrada
    prob_melhor: float = 0.0

    # -- métricas exatas, disponíveis apenas no método "diagonal" ----------
    #: Quantos reinícios independentes do otimizador clássico foram feitos
    n_reinicios: int = 1
    #: Índice do reinício que produziu este resultado
    reinicio_vencedor: int = 0
    #: Valor esperado de cada reinício, para medir a dispersão do otimizador
    valores_por_reinicio: list[float] = field(default_factory=list)
    #: Custo mínimo global do QUBO (de ``energias_qubo``), quando conhecido
    custo_minimo_global: float | None = None
    #: Probabilidade EXATA de amostrar uma solução de custo mínimo global
    prob_otimo: float | None = None
    #: Quantas configurações distintas atingem o mínimo global (degenerescência)
    n_estados_otimos: int = 0
    #: ``prob_otimo / 2^-n`` — quantas vezes o QAOA amplifica o ótimo em
    #: relação a sortear uma configuração uniformemente. Com k ótimos
    #: degenerados, a amplificação sobre sorteio uniforme é este valor / k.
    amplificacao: float | None = None
    #: Melhor custo no SUPORTE da distribuição — limite otimista, equivalente
    #: a medir infinitas vezes. ``melhor_custo`` é o valor realista, obtido
    #: com ``shots`` medições; a distância entre os dois mostra quanto o
    #: orçamento finito de medições custa ao método.
    custo_melhor_suporte: float | None = None
    #: True se as ``shots`` medições encontraram uma solução de custo ótimo
    shots_acertaram_otimo: bool | None = None
    #: Como o valor esperado foi calculado: "diagonal" ou "aer"
    metodo_valor_esperado: str = "diagonal"

    def resumo(self) -> str:
        extra = ""
        if self.prob_otimo is not None:
            extra = f" | P(ótimo)={self.prob_otimo:.2e} | ampl={self.amplificacao:.1f}x"
        return (
            f"QAOA(p={self.p}) | custo={self.melhor_custo:10.2f} | "
            f"qubits={self.n_qubits:3d} | profundidade={self.profundidade_circuito:5d} | "
            f"iterações={self.n_iteracoes:4d} | tempo={self.tempo_s:7.2f}s{extra}"
        )


def construir_hamiltoniano(modelo: ModeloQUBO) -> "SparsePauliOp":
    """Monta o Hamiltoniano de custo como soma de operadores de Pauli.

    O termo constante é deliberadamente omitido: ele desloca todas as energias
    igualmente e não influencia a otimização dos parâmetros. Os custos
    reportados são recalculados a partir do QUBO, de modo que essa omissão não
    afeta os resultados.
    """
    if not QISKIT_DISPONIVEL:
        raise ImportError(
            "Qiskit não encontrado. Instale com: python -m pip install qiskit"
        )

    h, J, _const = qubo_para_ising(modelo)
    n = modelo.n_variaveis
    termos: list[tuple[str, float]] = []

    for a in range(n):
        if h[a] != 0.0:
            pauli = ["I"] * n
            # Qiskit indexa os qubits da direita para a esquerda (little-endian)
            pauli[n - 1 - a] = "Z"
            termos.append(("".join(pauli), float(h[a])))

    for a in range(n):
        for b in range(a + 1, n):
            if J[a, b] != 0.0:
                pauli = ["I"] * n
                pauli[n - 1 - a] = "Z"
                pauli[n - 1 - b] = "Z"
                termos.append(("".join(pauli), float(J[a, b])))

    return SparsePauliOp.from_list(termos)


def _bitstring_para_vetor(bitstring: str, n: int) -> np.ndarray:
    """Converte a chave de contagem do Qiskit num vetor ``x`` na ordem do modelo."""
    return np.array([int(bitstring[n - 1 - a]) for a in range(n)], dtype=int)


def mixer_xy(modelo: ModeloQUBO, topologia: str = "anel") -> "SparsePauliOp":
    r"""Mixer que **preserva** a restrição de alocação única.

    O problema
    ----------
    O mixer padrão do QAOA é :math:`B = \sum_a X_a`, e :math:`X_a` inverte o
    bit ``a``. Como um período válido tem exatamente um bit em 1, inverter
    qualquer bit desse período ou apaga a única cultura ou acende uma segunda:
    **todo vizinho de um plano válido é inválido**. O subespaço viável tem
    ``n_culturas ** n_periodos`` planos de um total de :math:`2^n` -- em 24
    qubits, 1.296 de 16.777.216 -- e o mixer X empurra a amplitude para fora
    dele a cada camada.

    A solução
    ---------
    Aqui o vizinho é obtido **trocando** dois bits do mesmo período em vez de
    inverter um. O operador que faz isso num par é
    :math:`(X_aX_b + Y_aY_b)/2`, e o mixer é a soma sobre os pares de culturas
    de cada (talhão, período). Esse operador comuta com o que conta os 1s de
    cada período, ou seja: o peso de Hamming é uma **quantidade conservada da
    evolução**, como a energia num sistema mecânico. Nada é filtrado nem
    descartado classicamente -- as amplitudes fora do subespaço viável
    simplesmente permanecem em zero, porque o operador é incapaz de colocar
    amplitude nelas.

    Os termos do mixer não comutam entre si, então ``exp(-i beta B)`` é
    realizado pelo Qiskit como uma aproximação de Trotter. Isso afeta a
    *dinâmica* da mistura, **não** a conservação: cada fator preserva o peso
    de Hamming exatamente, logo qualquer produto deles também preserva. A
    viabilidade é exata; só a mistura é aproximada.

    Parâmetros
    ----------
    topologia:
        ``"anel"`` liga cada cultura à seguinte, fechando o ciclo
        (``n_culturas`` pares por período). ``"completo"`` liga todos os pares
        (``n_culturas*(n_culturas-1)/2``), o que mistura melhor e custa mais
        portas. Ambas são irredutíveis dentro do subespaço viável, isto é:
        qualquer plano válido alcança qualquer outro por uma sequência de
        trocas.
    """
    if not QISKIT_DISPONIVEL:  # pragma: no cover
        raise RuntimeError("Qiskit não está instalado.")

    termos: list[tuple[str, list[int], float]] = []
    for t_per in range(modelo.n_periodos):
        for j in range(modelo.n_talhoes):
            qubits = [modelo.indice(i, j, t_per)
                      for i in range(modelo.n_culturas)]
            if topologia == "anel":
                pares = [(qubits[k], qubits[(k + 1) % len(qubits)])
                         for k in range(len(qubits))]
                if len(qubits) == 2:   # o anel de 2 duplicaria o mesmo par
                    pares = [(qubits[0], qubits[1])]
            elif topologia == "completo":
                pares = [(a, b) for k, a in enumerate(qubits)
                         for b in qubits[k + 1:]]
            else:
                raise ValueError(
                    f"topologia desconhecida: {topologia!r} "
                    f"(use 'anel' ou 'completo')"
                )
            for a, b in pares:
                termos.append(("XX", [a, b], 0.5))
                termos.append(("YY", [a, b], 0.5))

    return SparsePauliOp.from_sparse_list(termos,
                                          num_qubits=modelo.n_variaveis)


def estado_inicial_onehot(modelo: ModeloQUBO) -> "QuantumCircuit":
    r"""Superposição uniforme sobre os planos válidos, e **só** sobre eles.

    O mixer XY conserva o peso de Hamming, o que é uma faca de dois gumes: ele
    nunca sai do subespaço viável, mas também nunca *entra* nele. Começar da
    superposição uniforme sobre os :math:`2^n` estados -- o padrão do QAOA --
    seria começar 99,99% em terreno inválido e ficar preso lá.

    Então o estado inicial dá peso igual às ``n_culturas`` opções de cada
    (talhão, período), e peso zero a tudo o mais. Em cada bloco de período
    isso é o chamado estado W,
    :math:`(|10\dots0\rangle + |01\dots0\rangle + \dots)/\sqrt{n_c}`.
    O produto dos blocos é a superposição uniforme sobre os
    ``n_culturas ** n_periodos`` planos válidos.
    """
    if not QISKIT_DISPONIVEL:  # pragma: no cover
        raise RuntimeError("Qiskit não está instalado.")

    nc = modelo.n_culturas
    vetor = np.zeros(2 ** nc)
    for i in range(nc):
        vetor[1 << i] = 1.0 / np.sqrt(nc)   # exatamente uma cultura: bit i

    qc = QuantumCircuit(modelo.n_variaveis)
    preparo = StatePreparation(vetor)
    for t_per in range(modelo.n_periodos):
        for j in range(modelo.n_talhoes):
            qc.append(preparo,
                      [modelo.indice(i, j, t_per) for i in range(nc)])
    return qc


def executar_qaoa(
    modelo: ModeloQUBO,
    p: int = 1,
    maxiter: int = 200,
    shots: int = 4096,
    seed: int | None = None,
    memoria_maxima_gb: float = 8.0,
    otimizador: str = "COBYLA",
    precisao: str = "double",
    n_reinicios: int = 5,
    metodo_valor_esperado: str = "diagonal",
    callback: Callable[[float], None] | None = None,
    threads: int = 8,
    mixer: str = "X",
    topologia_xy: str = "anel",
) -> ResultadoQAOA:
    """Executa o QAOA sobre o QUBO informado, em simulador por vetor de estado.

    Parameters
    ----------
    modelo:
        Instância do problema.
    p:
        Número de camadas do ansatz. Mais camadas aumentam a qualidade teórica
        da aproximação e também a profundidade do circuito e o número de
        parâmetros a otimizar.
    maxiter:
        Limite de iterações do otimizador clássico.
    shots:
        Amostras extraídas do estado final para decodificar a solução.
    seed:
        Semente para a inicialização dos parâmetros e para a amostragem.
    memoria_maxima_gb:
        Guarda de segurança. A execução é recusada se a simulação exigir mais
        memória do que este limite, evitando esgotar a RAM da máquina.
    otimizador:
        Método aceito por ``scipy.optimize.minimize``. COBYLA é o padrão por
        não exigir gradientes, indisponíveis de forma barata aqui.
    precisao:
        ``"double"`` (padrão) ou ``"single"``. Precisão simples corta a memória
        do vetor de estado pela metade, ao custo de erro numérico maior no
        valor esperado — aceitável aqui, pois a decisão final é tomada pelo
        custo QUBO exato de cada configuração, não pelo valor esperado.
    n_reinicios:
        Quantos pontos de partida independentes o otimizador clássico tenta,
        ficando com o melhor. O primeiro é a inicialização adiabática; os
        demais são sorteados. Com um único início o COBYLA trava em mínimos
        locais e a qualidade oscila sem padrão com ``p``.
    metodo_valor_esperado:
        ``"diagonal"`` (padrão) calcula ``<custo> = Σ |ψ_i|² E_i`` a partir do
        vetor de estado e das energias pré-calculadas. Como o Hamiltoniano de
        custo é diagonal, o resultado é idêntico ao do Aer — e dá de brinde a
        **distribuição de probabilidade exata**, de onde saem ``prob_otimo`` e
        ``amplificacao`` sem ruído de amostragem.

        ``"aer"`` usa ``EstimatorV2``/``SamplerV2`` do Qiskit Aer, com
        amostragem real por shots. É o caminho de referência, e
        :func:`validar_valor_esperado` confere que os dois concordam.

    Notes
    -----
    Toda a simulação é **ideal, sem ruído**: não há modelo de decoerência nem
    erro de porta. As probabilidades exatas reportadas valem para esse regime
    e são um limite superior do que se observaria em hardware real NISQ.

    Raises
    ------
    ImportError
        Se o Qiskit ou o Qiskit Aer não estiverem instalados.
    MemoryError
        Se a instância exceder ``memoria_maxima_gb``.
    """
    if not QISKIT_DISPONIVEL:
        raise ImportError(
            "Qiskit/Qiskit Aer não encontrados. Instale com: "
            "python -m pip install qiskit qiskit-aer"
        )

    n = modelo.n_variaveis
    memoria = estimar_memoria_gb(n, precisao)
    if memoria > memoria_maxima_gb:
        raise MemoryError(
            f"A instância exige {n} qubits (~{memoria:.1f} GB para o vetor de "
            f"estado em precisão {precisao}), acima do limite de "
            f"{memoria_maxima_gb:.1f} GB. Reduza o cenário, use "
            'precisao="single" ou eleve memoria_maxima_gb se a máquina '
            "comportar."
        )

    t_inicio = time.perf_counter()
    rng = np.random.default_rng(seed)

    hamiltoniano = construir_hamiltoniano(modelo)
    _h, _J, const_ising = qubo_para_ising(modelo)
    if mixer.upper() == "XY":
        # Mixer e estado inicial confinam a evolução ao subespaço viável.
        # Os dois andam juntos: o mixer não sai de onde está, e o estado
        # inicial é o que o coloca dentro da região que interessa.
        ansatz = QAOAAnsatz(
            cost_operator=hamiltoniano,
            reps=p,
            mixer_operator=mixer_xy(modelo, topologia_xy),
            initial_state=estado_inicial_onehot(modelo),
        )
    elif mixer.upper() == "X":
        ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=p)
    else:
        raise ValueError(f"mixer desconhecido: {mixer!r} (use 'X' ou 'XY')")
    ansatz = ansatz.decompose(reps=4)

    opcoes_aer = {
        "backend_options": {
            "method": "statevector",
            "precision": precisao,
            "seed_simulator": seed,
            "max_parallel_threads": threads,
        }
    }

    # Os coeficientes do Hamiltoniano são grandes (as penalidades chegam à casa
    # das centenas), o que colocaria os γ ótimos numa escala minúscula e
    # dificultaria a convergência do otimizador. Normalizar o valor esperado
    # devolve os parâmetros a uma escala O(1); como isso é uma reescala
    # monotônica, o mínimo não se desloca.
    escala = float(np.max(np.abs(hamiltoniano.coeffs.real)))
    escala = escala if escala > 0 else 1.0

    # --- como calcular o valor esperado -----------------------------------
    energias: np.ndarray | None = None
    if metodo_valor_esperado == "diagonal":
        energias = energias_qubo(modelo, limite_gb=memoria_maxima_gb)
        # Oito threads, e nao todas as 32 logicas. A simulacao e limitada pela
        # banda de memoria, nao pelo processamento: medido em 24 qubits, o
        # tempo cai de 11,03 s com 1 thread para 3,14 s com 8, e volta a
        # SUBIR para 3,65 s usando todas. Alem de 8 threads a disputa pela
        # memoria custa mais do que o paralelismo rende. O valor 0 devolve o
        # comportamento padrao do Aer, que e usar todas.
        backend_sv = AerSimulator(method="statevector", precision=precisao,
                                  max_parallel_threads=threads)
        circuito_sv = ansatz.copy()
        circuito_sv.save_statevector()
        circuito_sv = transpile(circuito_sv, backend_sv)
        nomes_par = list(ansatz.parameters)

        def estado(parametros: np.ndarray) -> np.ndarray:
            ligacoes = [{q: [v] for q, v in zip(nomes_par, parametros)}]
            res = backend_sv.run(circuito_sv, parameter_binds=ligacoes).result()
            return np.asarray(res.get_statevector())

        def valor_esperado_de(parametros: np.ndarray) -> float:
            psi = estado(parametros)
            return float((np.abs(psi) ** 2) @ energias)

        # O custo QUBO já inclui o offset, então não há constante a recompor.
        def para_escala_qubo(v: float) -> float:
            return v * escala

    elif metodo_valor_esperado == "aer":
        estimator = EstimatorV2(options=opcoes_aer)

        def valor_esperado_de(parametros: np.ndarray) -> float:
            job = estimator.run([(ansatz, hamiltoniano, parametros)])
            # O Hamiltoniano omite a constante da conversão Ising; somá-la de
            # volta deixa o valor na mesma escala do custo QUBO.
            return float(job.result()[0].data.evs) + const_ising

        def para_escala_qubo(v: float) -> float:
            return v * escala

    else:
        raise ValueError(
            f"metodo_valor_esperado desconhecido: {metodo_valor_esperado!r}. "
            "Use 'diagonal' ou 'aer'."
        )

    # --- laço híbrido, com múltiplos reinícios ----------------------------
    # Um único ponto de partida faz o COBYLA travar em mínimos locais, e a
    # qualidade passa a oscilar sem padrão com p — chegou-se a observar p=2
    # pior que amostragem aleatória. Vários reinícios e o melhor entre eles
    # tornam o resultado reprodutível. É prática padrão em QAOA e precisa ser
    # declarada em Materiais e Métodos.
    historico: list[float] = []

    def funcao_custo(parametros: np.ndarray) -> float:
        valor = valor_esperado_de(parametros) / escala
        historico.append(valor)
        if callback is not None:
            # Chamado a cada avaliacao. Em 28 qubits uma avaliacao leva mais de
            # um minuto, logo este e o unico ponto do codigo onde da para
            # mostrar que a execucao esta viva. O valor informado esta na
            # escala do QUBO, para ser legivel por quem acompanha.
            callback(para_escala_qubo(valor))
        return valor

    melhor_opt = None
    reinicio_vencedor = 0
    valores_por_reinicio: list[float] = []

    for r in range(max(1, n_reinicios)):
        if r == 0:
            # Inicialização adiabática: γ crescente e β decrescente ao longo
            # das camadas, imitando uma evolução adiabática discretizada.
            iniciais = np.concatenate([np.linspace(0.6, 0.1, p),
                                       np.linspace(0.1, 0.6, p)])
            if iniciais.size != ansatz.num_parameters:
                iniciais = rng.uniform(0, np.pi, ansatz.num_parameters)
        else:
            iniciais = rng.uniform(0, np.pi, ansatz.num_parameters)

        opt = minimize(
            funcao_custo, iniciais, method=otimizador,
            options={"maxiter": maxiter},
        )
        valores_por_reinicio.append(para_escala_qubo(float(opt.fun)))
        if melhor_opt is None or opt.fun < melhor_opt.fun:
            melhor_opt = opt
            reinicio_vencedor = r

    assert melhor_opt is not None

    # --- leitura da solução a partir do estado final -----------------------
    resultado_extra: dict = {}
    if metodo_valor_esperado == "diagonal":
        assert energias is not None
        psi = estado(melhor_opt.x)
        probabilidades = np.abs(psi) ** 2

        # Distribuição exata: nada aqui é estimado por amostragem.
        custo_minimo_global = float(energias.min())
        mascara_otimos = energias <= custo_minimo_global + 1e-9
        prob_otimo = float(probabilidades[mascara_otimos].sum())
        n_otimos = int(np.count_nonzero(mascara_otimos))

        # Melhor solução no SUPORTE da distribuição — um limite superior
        # otimista: é o melhor que se obteria com medições ilimitadas.
        suporte = probabilidades > 1.0 / (shots * 1000.0)
        if not suporte.any():
            suporte = probabilidades >= probabilidades.max()
        indices_suporte = np.flatnonzero(suporte)
        idx_suporte = int(indices_suporte[np.argmin(energias[indices_suporte])])
        custo_suporte = float(energias[idx_suporte])

        # Resultado POR SHOTS, amostrando a distribuição exata. É este que vira
        # ``melhor_custo``, por duas razões: é o que uma execução real com
        # orçamento finito de medições obteria, e é o que se compara de forma
        # justa com o AG, que também reporta a melhor solução que de fato
        # avaliou. Usar o suporte como resultado principal seria otimista —
        # chega-se a ter o ótimo no suporte com probabilidade de 1e-4, isto é,
        # invisível em 4096 medições.
        rng_shots = np.random.default_rng(seed)
        amostras = rng_shots.choice(
            probabilidades.size, size=shots,
            p=probabilidades / probabilidades.sum(),
        )
        custos_amostrados = energias[amostras]
        melhor_idx = int(amostras[np.argmin(custos_amostrados)])
        melhor_custo = float(energias[melhor_idx])
        melhor_x = np.array(
            [(melhor_idx >> a) & 1 for a in range(n)], dtype=int
        )
        mascara_melhor = energias <= melhor_custo + 1e-9
        prob_melhor = float(probabilidades[mascara_melhor].sum())

        resultado_extra = dict(
            custo_minimo_global=custo_minimo_global,
            prob_otimo=prob_otimo,
            n_estados_otimos=n_otimos,
            amplificacao=prob_otimo * (2.0**n),
            custo_melhor_suporte=custo_suporte,
            shots_acertaram_otimo=bool(
                melhor_custo <= custo_minimo_global + 1e-9
            ),
        )
    else:
        # Caminho Aer de referência: amostragem real, como em hardware.
        circuito_medido = ansatz.copy()
        circuito_medido.measure_all()
        backend = AerSimulator(method="statevector", precision=precisao,
                               max_parallel_threads=threads)
        circuito_medido = transpile(circuito_medido, backend)

        sampler = SamplerV2(options=opcoes_aer)
        job = sampler.run([(circuito_medido, melhor_opt.x)], shots=shots)
        contagens = job.result()[0].data.meas.get_counts()

        melhor_x = None
        melhor_custo = np.inf
        contagem_melhor = 0
        for bitstring, contagem in contagens.items():
            x = _bitstring_para_vetor(bitstring, n)
            custo = modelo.avaliar(x)
            if custo < melhor_custo - 1e-9:
                melhor_custo, melhor_x, contagem_melhor = custo, x, contagem
            elif abs(custo - melhor_custo) <= 1e-9:
                # Configurações distintas podem empatar no mesmo custo (o QUBO
                # tem ótimos degenerados). Somar as contagens evita subestimar
                # a probabilidade de amostrar uma solução de custo ótimo.
                contagem_melhor += contagem
        assert melhor_x is not None
        melhor_custo = float(melhor_custo)
        prob_melhor = contagem_melhor / shots

    return ResultadoQAOA(
        melhor_x=melhor_x,
        melhor_custo=melhor_custo,
        valor_esperado=para_escala_qubo(float(melhor_opt.fun)),
        parametros_otimos=np.asarray(melhor_opt.x),
        n_qubits=n,
        p=p,
        profundidade_circuito=ansatz.depth(),
        n_portas=sum(ansatz.count_ops().values()),
        n_iteracoes=int(getattr(melhor_opt, "nit", len(historico))),
        n_avaliacoes=len(historico),
        tempo_s=time.perf_counter() - t_inicio,
        historico=[para_escala_qubo(v) for v in historico],
        prob_melhor=prob_melhor,
        n_reinicios=max(1, n_reinicios),
        reinicio_vencedor=reinicio_vencedor,
        valores_por_reinicio=valores_por_reinicio,
        metodo_valor_esperado=metodo_valor_esperado,
        **resultado_extra,
    )


def validar_valor_esperado(
    modelo: ModeloQUBO, p: int = 2, seed: int | None = 0
) -> float:
    """Valida o método diagonal contra o Aer, no mesmo conjunto de parâmetros.

    Condição de aceite: diferença absoluta < 1e-6 entre os dois valores
    esperados. Esta checagem existe porque o método diagonal reimplementa o
    cálculo do valor esperado fora do Qiskit; sem ela, um erro de convenção de
    índice passaria despercebido e contaminaria todos os resultados.

    Returns
    -------
    float
        Maior diferença absoluta observada entre os dois métodos.
    """
    if not QISKIT_DISPONIVEL:
        raise ImportError("Qiskit/Qiskit Aer não encontrados.")

    n = modelo.n_variaveis
    hamiltoniano = construir_hamiltoniano(modelo)
    _h, _J, const_ising = qubo_para_ising(modelo)
    ansatz = QAOAAnsatz(cost_operator=hamiltoniano, reps=p).decompose(reps=3)

    energias = energias_qubo(modelo)
    backend = AerSimulator(method="statevector")
    circuito = ansatz.copy()
    circuito.save_statevector()
    circuito = transpile(circuito, backend)
    nomes_par = list(ansatz.parameters)
    estimator = EstimatorV2(
        options={"backend_options": {"method": "statevector"}}
    )

    rng = np.random.default_rng(seed)
    maior_diferenca = 0.0
    for _ in range(5):
        parametros = rng.uniform(0, np.pi, ansatz.num_parameters)
        res = backend.run(
            circuito,
            parameter_binds=[{q: [v] for q, v in zip(nomes_par, parametros)}],
        ).result()
        psi = np.asarray(res.get_statevector())
        via_diagonal = float((np.abs(psi) ** 2) @ energias)

        job = estimator.run([(ansatz, hamiltoniano, parametros)])
        via_aer = float(job.result()[0].data.evs) + const_ising

        maior_diferenca = max(maior_diferenca, abs(via_diagonal - via_aer))

    return maior_diferenca


if __name__ == "__main__":
    print("Viabilidade da simulação por vetor de estado\n")
    print(tabela_de_viabilidade())

    print("\n\nValidação da conversão QUBO -> Ising")
    print("(a energia de Ising deve reproduzir exatamente o custo do QUBO)\n")
    from parametros import CENARIOS_POR_TAMANHO

    for n_cult in sorted(CENARIOS_POR_TAMANHO):
        modelo = ModeloQUBO(culturas=CENARIOS_POR_TAMANHO[n_cult])
        erro = verificar_conversao(modelo)
        estado = "OK" if erro < 1e-6 else "FALHOU"
        print(
            f"  {modelo.n_variaveis:3d} qubits ({n_cult} culturas x 4 períodos): "
            f"erro máximo = {erro:.2e}  [{estado}]"
        )

    if not QISKIT_DISPONIVEL:
        print(
            "\n\nQiskit/Qiskit Aer não instalados — execução do QAOA ignorada."
            "\nInstale com:  python -m pip install qiskit qiskit-aer"
        )
        raise SystemExit(0)

    # Instância pequena o bastante para rodar em qualquer máquina.
    modelo = ModeloQUBO(n_periodos=2)
    print(f"\n\nExecutando o QAOA em uma instância de {modelo.n_variaveis} qubits...\n")

    _x_otimo, custo_otimo = modelo.otimo_viavel()
    print(f"Melhor plano viável (enumeração): {custo_otimo:.2f}\n")

    resultado = None
    for p in (1, 2, 3):
        resultado = executar_qaoa(modelo, p=p, seed=42)
        desvio = 100.0 * (resultado.melhor_custo - custo_otimo) / abs(custo_otimo)
        print(f"  {resultado.resumo()} | desvio={desvio:+.2f}%")

    assert resultado is not None
    print("\nMelhor plano encontrado pelo QAOA:\n")
    print(modelo.relatorio(resultado.melhor_x))
