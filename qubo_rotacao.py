"""
qubo_rotacao.py
===============

Modelagem do problema de rotação de culturas como QUBO
(*Quadratic Unconstrained Binary Optimization*).

Este módulo é o **núcleo compartilhado** do trabalho: tanto o algoritmo genético
(:mod:`algoritmo_genetico`) quanto o QAOA (:mod:`qaoa_rotacao`) otimizam
exatamente a mesma matriz Q construída aqui. É essa identidade que torna a
comparação entre os dois paradigmas justa — a única diferença entre eles é a
estratégia de busca, nunca o problema.

Formulação
----------
Variáveis de decisão binárias:

    x[i,j,t] = 1  se a cultura i é alocada ao talhão j no período t
             = 0  caso contrário

A função objetivo tem a forma canônica QUBO

    custo(x) = x^T Q x + offset

com Q armazenada em forma **triangular superior incluindo a diagonal**:

    x^T Q x = Σ_a Q[a,a]·x_a  +  Σ_{a<b} Q[a,b]·x_a·x_b

(o que é válido porque x_a ∈ {0,1} implica x_a² = x_a, permitindo colapsar os
coeficientes lineares na diagonal).

Termos da função objetivo
-------------------------
1. **Custo de nitrogênio** (linear + quadrático) — o termo agronomicamente
   central. O coeficiente linear assume o pior caso (antecessora gramínea) e o
   termo quadrático desconta o *crédito de N* quando a cultura do período
   anterior, no mesmo talhão, é uma leguminosa. É assim que o modelo "aprende"
   a preferir sequências leguminosa → gramínea.
2. **Balanço de nitrogênio** (quadrático) — penaliza o desvio quadrático do
   saldo de N acumulado ao longo do horizonte, onde o saldo de cada cultivo é
   ``reposição − exportação`` (ver :meth:`Cultura.saldo_n`). Empurra o plano
   para fechar o ciclo em equilíbrio, penalizando tanto o esgotamento do solo
   quanto a superadubação.
3. **Penalidade de alocação única** — exatamente uma cultura por talhão/período.
4. **Penalidade fitossanitária** — desestimula repetir a mesma cultura, e em
   menor grau a mesma família botânica, em períodos consecutivos; e repetir a
   mesma cultura na mesma estação de anos consecutivos (t e t+2).
5. **Penalidade de estação** — desestimula alocar uma cultura de verão num
   período de inverno e vice-versa.

Os termos 3 a 5 são as restrições agronômicas incorporadas como penalização,
conforme prevê a formulação QUBO.

Composição versus sequência
---------------------------
Vale distinguir o que cada termo é capaz de enxergar, porque isso é uma
limitação da própria forma QUBO e não uma escolha de modelagem:

- Os termos de **composição** dependem só de *quais* culturas entram no plano,
  não da ordem: o custo linear de N e o balanço de N. O balanço é
  ``λ(Σ saldo)²``; um saldo que dependesse da antecessora já seria quadrático,
  e elevá-lo ao quadrado produziria termos de quarta ordem, fora da forma
  QUBO. Permutar as culturas entre os períodos não muda o balanço.
- Os termos de **sequência** dependem da ordem: o crédito de N e as
  penalidades fitossanitárias. São naturalmente quadráticos, por ligarem pares
  de períodos adjacentes.

A tese do trabalho vive na camada de sequência, e é por isso que ``custo_n``
vale 3,0 (ver :class:`Pesos`): é o que faz o crédito de N superar a penalidade
fitossanitária e efetivamente decidir a ordem.

Fósforo e potássio
------------------
P e K **não** entram na função objetivo (pesos zero por padrão). A adubação de
manutenção é uma constante por cultura, logo é informação de composição pura, e
com os pesos anteriores respondia por 54% do valor da função objetivo sem
carregar nenhuma informação de rotação. Seguem calculados e reportados como
métrica agronômica dos planos, em :meth:`ModeloQUBO.adubacao_pk_total`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from parametros import (
    ANTECESSORA_GRAMINEA,
    ANTECESSORA_LEGUMINOSA,
    ANTECESSORA_POUSIO,
    CENARIO_BASE,
    ESTACAO_INVERNO,
    ESTACAO_VERAO,
    FAIXA_MO_PADRAO,
    Cultura,
)


@dataclass
class Pesos:
    """Pesos e penalidades da função objetivo.

    Nitrogênio é o único nutriente na função objetivo
    ---------------------------------------------------
    ``custo_n`` pondera a dose de N (kg/ha) convertendo-a em custo escalar. O
    valor 3,0 é deliberado: com peso 1,0 o *crédito de N* — os 20 kg/ha que
    uma leguminosa antecedente economiza, e o único dado do Manual que enxerga
    a ordem das culturas — ficava abaixo da penalidade fitossanitária de mesma
    família (50) e não decidia nada. Com peso 3,0 o crédito vale 60 e passa a
    comandar a escolha da sequência, que é a tese do trabalho.

    ``custo_p2o5`` e ``custo_k2o`` são **zero por padrão**: fósforo e potássio
    saíram da função objetivo. A adubação de manutenção de P e K é uma
    constante por cultura — não depende da sequência — e com peso 1,0/0,45/0,35
    ela respondia por 54% do valor da função objetivo sem carregar nenhuma
    informação de rotação, achatando os termos que carregam. P e K seguem
    sendo calculados e reportados como métrica agronômica dos planos (ver
    :meth:`ModeloQUBO.adubacao_pk_total`); os campos continuam aqui para que
    reativá-los seja um ajuste de peso, não uma mudança de código.

    Balanço de nitrogênio
    ---------------------
    ``penal_balanco_n`` pondera o desvio quadrático do saldo de N acumulado no
    horizonte. Ver o termo 5 no docstring do módulo.

    Penalidades
    -----------
    Precisam ser maiores que qualquer economia que a violação da restrição
    correspondente poderia gerar; caso contrário o otimizador "compra" a
    violação. :meth:`ModeloQUBO.verificar_penalidades` checa isso.
    """

    custo_n: float = 3.00
    custo_p2o5: float = 0.00
    custo_k2o: float = 0.00

    #: Peso do desvio quadrático do balanço de N. Calibrado para que a
    #: diferença de balanço entre planos típicos fique na mesma ordem de
    #: grandeza da diferença de custo de N entre eles.
    penal_balanco_n: float = 0.03

    penal_alocacao_unica: float = 600.0
    penal_repeticao_cultura: float = 150.0
    #: Deve superar ``custo_n × maior dose de N`` (ver
    #: :meth:`ModeloQUBO.verificar_penalidades`). Com ``custo_n = 3,0`` e dose
    #: máxima de 75 kg/ha, o limiar é 225 — por isso 300, e não os 200 que
    #: bastavam quando ``custo_n`` valia 1,0. Com 200 o ótimo do cenário de 5
    #: culturas passava a ser monocultura sazonal de soja: o plano pagava a
    #: penalidade e ainda saía mais barato, por a soja não demandar N.
    penal_repeticao_estacional: float = 300.0
    penal_mesma_familia: float = 50.0
    penal_estacao: float = 300.0


@dataclass
class ModeloQUBO:
    """Constrói e avalia o QUBO do problema de rotação de culturas.

    Parameters
    ----------
    culturas:
        Culturas candidatas (ver :mod:`parametros`).
    n_talhoes, n_periodos:
        Dimensões do cenário. O número de variáveis binárias — e portanto de
        qubits necessários ao QAOA — é ``len(culturas) * n_talhoes *
        n_periodos``.
    pesos:
        Pesos da função objetivo. Usa :class:`Pesos` padrão se omitido.
    faixa_mo:
        Faixa de matéria orgânica do solo (afeta todas as doses de N).
    antecessora_inicial:
        Tipo de cobertura que ocupava a área antes do primeiro período.

    Notes
    -----
    Não há mais parâmetro ``meta_comercial``. O termo que obrigava um número
    mínimo de cultivos comerciais foi removido por ser **inerte**: todas as
    culturas de verão do catálogo são comerciais, e como a penalidade de
    estação já impede plantar cobertura no verão, qualquer plano admissível
    tem automaticamente um cultivo comercial em cada período de verão.
    Verificou-se que metas de 1, 2 ou 3 produziam exatamente o mesmo plano
    ótimo, alterando apenas o valor da função objetivo.

    Removê-lo também deixou o circuito do QAOA bem mais raso: o termo acoplava
    **todas** as variáveis comerciais entre si, o que no cenário de 8 culturas
    seriam 45 pares de qubits.
    """

    culturas: list[Cultura] = field(default_factory=lambda: list(CENARIO_BASE))
    n_talhoes: int = 1
    n_periodos: int = 4
    pesos: Pesos = field(default_factory=Pesos)
    faixa_mo: str = FAIXA_MO_PADRAO
    antecessora_inicial: str = ANTECESSORA_POUSIO

    # Preenchidos em __post_init__
    Q: np.ndarray = field(init=False, repr=False)
    offset: float = field(init=False, repr=False)
    estacoes: list[str] = field(init=False, repr=False)

    # -- construção --------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.culturas:
            raise ValueError("É necessária ao menos uma cultura.")
        if self.n_talhoes < 1 or self.n_periodos < 1:
            raise ValueError("n_talhoes e n_periodos devem ser >= 1.")

        # O horizonte alterna verão e inverno, começando no verão.
        self.estacoes = [
            ESTACAO_VERAO if t % 2 == 0 else ESTACAO_INVERNO
            for t in range(self.n_periodos)
        ]

        self.Q, self.offset = self._construir_matriz()

    @property
    def n_culturas(self) -> int:
        return len(self.culturas)

    @property
    def n_variaveis(self) -> int:
        """Número de variáveis binárias (= número de qubits para o QAOA)."""
        return self.n_culturas * self.n_talhoes * self.n_periodos

    def indice(self, i: int, j: int, t: int) -> int:
        """Mapeia (cultura, talhão, período) para o índice da variável binária."""
        return i + self.n_culturas * (j + self.n_talhoes * t)

    def decodificar_indice(self, a: int) -> tuple[int, int, int]:
        """Operação inversa de :meth:`indice`."""
        i = a % self.n_culturas
        resto = a // self.n_culturas
        j = resto % self.n_talhoes
        t = resto // self.n_talhoes
        return i, j, t

    def _construir_matriz(self) -> tuple[np.ndarray, float]:
        n = self.n_variaveis
        Q = np.zeros((n, n), dtype=float)
        offset = 0.0
        p = self.pesos

        def somar_quadratico(a: int, b: int, valor: float) -> None:
            """Acumula em Q[a,b] respeitando a convenção triangular superior."""
            if a == b:
                Q[a, a] += valor
            elif a < b:
                Q[a, b] += valor
            else:
                Q[b, a] += valor

        # --- Termos 1, 2 e 6: custos de nutriente e adequação de estação ---
        for t in range(self.n_periodos):
            for j in range(self.n_talhoes):
                for i, cultura in enumerate(self.culturas):
                    a = self.indice(i, j, t)

                    # (1) N — linha de base no pior caso.
                    # No primeiro período a antecessora é dada pelo estado
                    # inicial da área; nos demais, assume-se gramínea e o
                    # crédito é descontado adiante pelo termo quadrático.
                    if t == 0:
                        antecessora = self.antecessora_inicial
                    else:
                        antecessora = ANTECESSORA_GRAMINEA
                    Q[a, a] += p.custo_n * cultura.demanda_n(
                        self.faixa_mo, antecessora
                    )

                    # (2) P e K — adubação de manutenção, independe da sequência
                    Q[a, a] += p.custo_p2o5 * cultura.manutencao_p2o5
                    Q[a, a] += p.custo_k2o * cultura.manutencao_k2o

                    # (6) Estação incompatível
                    if cultura.estacao != self.estacoes[t]:
                        Q[a, a] += p.penal_estacao

        # --- Termo 1 (parte quadrática): crédito de N da antecessora -------
        # Para cada par (cultura no período t-1, cultura no período t) do mesmo
        # talhão, ajusta-se o custo de N da cultura sucessora em função da
        # família da antecessora. Esta é a única parte do modelo que enxerga a
        # ORDEM das culturas, e é o que caracteriza o problema como rotação.
        for t in range(1, self.n_periodos):
            for j in range(self.n_talhoes):
                for i_ant, ant in enumerate(self.culturas):
                    a = self.indice(i_ant, j, t - 1)

                    if ant.familia == "leguminosa":
                        tipo_ant = ANTECESSORA_LEGUMINOSA
                    elif ant.familia == "graminea":
                        tipo_ant = ANTECESSORA_GRAMINEA
                    else:
                        tipo_ant = ANTECESSORA_POUSIO

                    for i_suc, suc in enumerate(self.culturas):
                        b = self.indice(i_suc, j, t)

                        base = suc.demanda_n(self.faixa_mo, ANTECESSORA_GRAMINEA)
                        real = suc.demanda_n(self.faixa_mo, tipo_ant)
                        ajuste = p.custo_n * (real - base)
                        if ajuste != 0.0:
                            somar_quadratico(a, b, ajuste)

        # --- Termo 4: penalidades fitossanitárias de sequência -------------
        for t in range(1, self.n_periodos):
            for j in range(self.n_talhoes):
                for i_ant, ant in enumerate(self.culturas):
                    a = self.indice(i_ant, j, t - 1)
                    for i_suc, suc in enumerate(self.culturas):
                        b = self.indice(i_suc, j, t)
                        if i_ant == i_suc:
                            somar_quadratico(a, b, p.penal_repeticao_cultura)
                        elif ant.familia == suc.familia:
                            somar_quadratico(a, b, p.penal_mesma_familia)

        # Repetição da mesma cultura na mesma estação de anos consecutivos
        # (períodos t e t+2). Sem este termo o modelo aceita monocultura
        # sazonal — soja em todos os verões, por exemplo — que não viola a
        # regra de períodos consecutivos mas é justamente o padrão que a
        # rotação de culturas existe para evitar, pelo acúmulo de patógenos e
        # pragas específicos da cultura (FAOConservationAgriculture).
        for t in range(2, self.n_periodos):
            for j in range(self.n_talhoes):
                for i in range(self.n_culturas):
                    a = self.indice(i, j, t - 2)
                    b = self.indice(i, j, t)
                    somar_quadratico(a, b, p.penal_repeticao_estacional)

        # --- Termo 3: exatamente uma cultura por talhão/período ------------
        # λ(Σ_i x_i − 1)² = λ(−Σ_i x_i + 2Σ_{i<i'} x_i x_i' + 1)
        lam = p.penal_alocacao_unica
        for t in range(self.n_periodos):
            for j in range(self.n_talhoes):
                indices = [self.indice(i, j, t) for i in range(self.n_culturas)]
                for a in indices:
                    Q[a, a] -= lam
                for a, b in itertools.combinations(indices, 2):
                    somar_quadratico(a, b, 2.0 * lam)
                offset += lam

        # --- Termo 2: balanço de nitrogênio do ciclo ------------------------
        # Penaliza o desvio quadrático do saldo de N acumulado no horizonte,
        # por talhão:
        #
        #     λ (Σ_a s_a x_a)²  com  s_a = saldo_n da cultura da variável a
        #
        # Expandindo, e usando x² = x para colapsar o quadrado na diagonal:
        #
        #     λ (Σ_a s_a x_a)² = λ[ Σ_a s_a² x_a + 2 Σ_{a<b} s_a s_b x_a x_b ]
        #
        # O alvo é saldo zero, então não há constante e nada vai para o offset.
        # Note que o somatório percorre todos os períodos de UM talhão: o saldo
        # é uma grandeza do solo, e superávit num talhão não compensa déficit
        # em outro, porque o solo não se move entre áreas.
        lam = p.penal_balanco_n
        if lam != 0.0:
            saldos = [c.saldo_n(self.faixa_mo) for c in self.culturas]
            for j in range(self.n_talhoes):
                grupo = [
                    (self.indice(i, j, t), saldos[i])
                    for t in range(self.n_periodos)
                    for i in range(self.n_culturas)
                ]
                for a, s_a in grupo:
                    Q[a, a] += lam * s_a * s_a
                for (a, s_a), (b, s_b) in itertools.combinations(grupo, 2):
                    if s_a != 0.0 and s_b != 0.0:
                        somar_quadratico(a, b, 2.0 * lam * s_a * s_b)

        return Q, offset

    # -- avaliação ---------------------------------------------------------

    def avaliar(self, x: np.ndarray) -> float:
        """Custo de uma única solução candidata."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_variaveis:
            raise ValueError(
                f"Vetor de tamanho {x.size}; esperado {self.n_variaveis}."
            )
        return float(x @ self.Q @ x) + self.offset

    def avaliar_populacao(self, X: np.ndarray) -> np.ndarray:
        """Custo de uma população inteira, de uma só vez.

        ``X`` tem forma ``(m, n_variaveis)``. A avaliação vetorizada é o que
        torna o algoritmo genético eficiente: uma geração inteira é avaliada
        com um único produto matricial.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_variaveis:
            raise ValueError(
                f"Esperado formato (m, {self.n_variaveis}); recebido {X.shape}."
            )
        return np.einsum("ij,ij->i", X @ self.Q, X) + self.offset

    # -- inspeção de soluções ---------------------------------------------

    def alocacao_valida(self, x: np.ndarray) -> bool:
        """True se cada talhão/período recebe exatamente uma cultura."""
        grade = np.asarray(x, dtype=int).reshape(
            self.n_periodos, self.n_talhoes, self.n_culturas
        )
        return bool(np.all(grade.sum(axis=2) == 1))

    def plano_de_rotacao(self, x: np.ndarray) -> list[list[str]]:
        """Traduz o vetor binário num plano legível ``[talhão][período]``."""
        grade = np.asarray(x, dtype=int).reshape(
            self.n_periodos, self.n_talhoes, self.n_culturas
        )
        plano: list[list[str]] = []
        for j in range(self.n_talhoes):
            linha: list[str] = []
            for t in range(self.n_periodos):
                escolhidas = np.flatnonzero(grade[t, j])
                if escolhidas.size == 1:
                    linha.append(self.culturas[escolhidas[0]].nome)
                elif escolhidas.size == 0:
                    linha.append("(vazio)")
                else:
                    nomes = "+".join(self.culturas[i].nome for i in escolhidas)
                    linha.append(f"(conflito: {nomes})")
            plano.append(linha)
        return plano

    def demanda_n_total(self, x: np.ndarray) -> float:
        """N mineral total (kg/ha) exigido pelo plano, somando talhões/períodos.

        Diferentemente de :meth:`avaliar`, devolve uma grandeza agronômica
        interpretável, sem pesos nem penalidades — é o número que se reporta na
        discussão dos resultados.
        """
        grade = np.asarray(x, dtype=int).reshape(
            self.n_periodos, self.n_talhoes, self.n_culturas
        )
        total = 0.0
        for j in range(self.n_talhoes):
            tipo_ant = self.antecessora_inicial
            for t in range(self.n_periodos):
                escolhidas = np.flatnonzero(grade[t, j])
                if escolhidas.size != 1:
                    continue
                cultura = self.culturas[escolhidas[0]]
                total += cultura.demanda_n(self.faixa_mo, tipo_ant)
                if cultura.familia == "leguminosa":
                    tipo_ant = ANTECESSORA_LEGUMINOSA
                elif cultura.familia == "graminea":
                    tipo_ant = ANTECESSORA_GRAMINEA
                else:
                    tipo_ant = ANTECESSORA_POUSIO
        return total

    def saldo_n_total(self, x: np.ndarray) -> float:
        """Saldo de N acumulado pelo plano (kg/ha), somando talhões/períodos.

        É a grandeza que o termo de balanço minimiza em valor absoluto. Zero
        significa ciclo fechado em equilíbrio; positivo, acúmulo no solo;
        negativo, extração líquida.
        """
        grade = np.asarray(x, dtype=int).reshape(
            self.n_periodos, self.n_talhoes, self.n_culturas
        )
        total = 0.0
        for t in range(self.n_periodos):
            for j in range(self.n_talhoes):
                escolhidas = np.flatnonzero(grade[t, j])
                if escolhidas.size != 1:
                    continue
                total += self.culturas[escolhidas[0]].saldo_n(self.faixa_mo)
        return total

    def adubacao_pk_total(self, x: np.ndarray) -> tuple[float, float]:
        """Adubação de manutenção de P2O5 e K2O exigida pelo plano (kg/ha).

        P e K ficaram **fora** da função objetivo (ver :class:`Pesos`), mas
        continuam sendo grandezas agronômicas do plano e entram no relatório de
        resultados: um plano que economiza nitrogênio às custas de muito
        fósforo não é necessariamente melhor, e o leitor precisa ver os dois
        números.
        """
        grade = np.asarray(x, dtype=int).reshape(
            self.n_periodos, self.n_talhoes, self.n_culturas
        )
        p2o5 = k2o = 0.0
        for t in range(self.n_periodos):
            for j in range(self.n_talhoes):
                escolhidas = np.flatnonzero(grade[t, j])
                if escolhidas.size != 1:
                    continue
                cultura = self.culturas[escolhidas[0]]
                p2o5 += cultura.manutencao_p2o5
                k2o += cultura.manutencao_k2o
        return p2o5, k2o

    def relatorio(self, x: np.ndarray) -> str:
        """Descrição textual completa de uma solução."""
        x = np.asarray(x, dtype=int).ravel()
        plano = self.plano_de_rotacao(x)
        largura = max(len(c.nome) for c in self.culturas) + 2
        p2o5, k2o = self.adubacao_pk_total(x)

        linhas = [
            f"Custo da função objetivo : {self.avaliar(x):.2f}",
            f"Alocação válida          : "
            f"{'sim' if self.alocacao_valida(x) else 'NÃO (solução inviável)'}",
            f"Demanda total de N       : {self.demanda_n_total(x):.0f} kg/ha",
            f"Saldo de N do ciclo      : {self.saldo_n_total(x):+.0f} kg/ha "
            f"(0 = equilibrio)",
            f"Adubacao P2O5 / K2O      : {p2o5:.0f} / {k2o:.0f} kg/ha "
            f"(fora do objetivo)",
            "",
        ]

        cabecalho = f"{'Talhão':<8}" + "".join(
            f"{f'T{t + 1} ({self.estacoes[t][:3]})':<{largura}}"
            for t in range(self.n_periodos)
        )
        linhas.append(cabecalho)
        linhas.append("-" * len(cabecalho))
        for j, linha in enumerate(plano):
            linhas.append(
                f"{j + 1:<8}" + "".join(f"{nome:<{largura}}" for nome in linha)
            )
        return "\n".join(linhas)

    # -- utilidades para os otimizadores ----------------------------------

    def para_qiskit(self) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
        """Exporta o QUBO no formato aceito pelo ``QuadraticProgram`` do Qiskit.

        Returns
        -------
        (linear, quadratico, offset)
            ``linear[a]`` e ``quadratico[(a, b)]`` com ``a < b``.
        """
        linear = {
            a: float(self.Q[a, a])
            for a in range(self.n_variaveis)
            if self.Q[a, a] != 0.0
        }
        quadratico = {
            (a, b): float(self.Q[a, b])
            for a in range(self.n_variaveis)
            for b in range(a + 1, self.n_variaveis)
            if self.Q[a, b] != 0.0
        }
        return linear, quadratico, self.offset

    def otimo_viavel(self, limite_combinacoes: int = 20_000_000) -> tuple[np.ndarray, float]:
        """Melhor solução **viável**, por enumeração exaustiva.

        Enumera apenas as soluções que respeitam a alocação única (uma cultura
        por talhão/período), o que reduz o espaço de 2^n para
        ``n_culturas^(n_talhoes·n_periodos)``. Para o cenário-base isso são
        4^8 = 65.536 combinações — instantâneo, e suficiente para conhecer o
        alvo agronomicamente correto e reportar o desvio real de cada
        metaheurística, em vez de compará-las apenas entre si.

        Atenção
        -------
        Este **não** é necessariamente o mínimo global do QUBO. O QUBO é
        irrestrito: se as penalidades estiverem subdimensionadas, existirão
        soluções inviáveis mais baratas, e uma metaheurística bem-sucedida as
        encontrará. Use :meth:`otimo_global` e :meth:`verificar_penalidades`
        para detectar essa situação.

        Raises
        ------
        ValueError
            Se o espaço viável exceder ``limite_combinacoes``.
        """
        n_slots = self.n_talhoes * self.n_periodos
        total = self.n_culturas**n_slots
        if total > limite_combinacoes:
            raise ValueError(
                f"Espaço viável com {total:.3g} combinações excede o limite de "
                f"{limite_combinacoes:.3g}. Use uma instância menor."
            )

        melhor_x: np.ndarray | None = None
        melhor_custo = np.inf
        lote_tamanho = 4096
        lote: list[np.ndarray] = []

        def processar(lote: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
            X = np.array(lote, dtype=float)
            custos = self.avaliar_populacao(X)
            k = int(np.argmin(custos))
            return X[k].astype(int), float(custos[k])

        for combo in itertools.product(range(self.n_culturas), repeat=n_slots):
            x = np.zeros(self.n_variaveis, dtype=float)
            for slot, i in enumerate(combo):
                j = slot % self.n_talhoes
                t = slot // self.n_talhoes
                x[self.indice(i, j, t)] = 1.0
            lote.append(x)

            if len(lote) >= lote_tamanho:
                cand_x, cand_c = processar(lote)
                if cand_c < melhor_custo:
                    melhor_x, melhor_custo = cand_x, cand_c
                lote = []

        if lote:
            cand_x, cand_c = processar(lote)
            if cand_c < melhor_custo:
                melhor_x, melhor_custo = cand_x, cand_c

        assert melhor_x is not None
        return melhor_x, melhor_custo

    def otimo_global(self, limite_qubits: int = 24) -> tuple[np.ndarray, float]:
        """Mínimo global do QUBO sobre **todo** o espaço {0,1}^n.

        Enumera as 2^n configurações, inclusive as inviáveis. Serve para
        validar a formulação: numa penalização bem dimensionada, o mínimo
        global deve coincidir com :meth:`otimo_viavel`. Se for menor, alguma
        penalidade está fraca demais e o modelo está premiando uma solução sem
        sentido agronômico.

        Raises
        ------
        ValueError
            Se ``n_variaveis`` exceder ``limite_qubits``.
        """
        n = self.n_variaveis
        if n > limite_qubits:
            raise ValueError(
                f"Enumerar 2^{n} configurações é inviável (limite: "
                f"2^{limite_qubits}). Use uma instância menor."
            )

        melhor_x: np.ndarray | None = None
        melhor_custo = np.inf
        lote_tamanho = 1 << 16
        pesos_bits = (1 << np.arange(n, dtype=np.uint64)).astype(np.uint64)

        for inicio in range(0, 1 << n, lote_tamanho):
            fim = min(inicio + lote_tamanho, 1 << n)
            inteiros = np.arange(inicio, fim, dtype=np.uint64)[:, None]
            X = ((inteiros & pesos_bits) > 0).astype(float)
            custos = self.avaliar_populacao(X)
            k = int(np.argmin(custos))
            if custos[k] < melhor_custo:
                melhor_custo = float(custos[k])
                melhor_x = X[k].astype(int)

        assert melhor_x is not None
        return melhor_x, melhor_custo

    def verificar_penalidades(self, exato: bool = False) -> list[str]:
        """Checa se as penalidades dominam os custos que deveriam proibir.

        Uma penalidade mal dimensionada faz o otimizador preferir uma solução
        inviável barata a uma viável correta — um erro silencioso que produz
        resultados sem sentido agronômico. Devolve a lista de avisos; lista
        vazia significa que os pesos estão consistentes.

        Parameters
        ----------
        exato:
            Se True, compara :meth:`otimo_viavel` com :meth:`otimo_global` por
            enumeração completa. É a checagem definitiva — as demais são
            heurísticas conservadoras que analisam um termo de cada vez e não
            capturam interações entre penalidades. Só é aplicável a instâncias
            pequenas.
        """
        avisos: list[str] = []
        p = self.pesos

        # Maior custo que uma alocação pode ter: se a penalidade de alocação
        # única não o superar, deixar o slot vazio compensa. Inclui a parcela
        # diagonal do termo de balanço (λ·saldo²), que também encarece ocupar
        # o slot.
        maior_custo_slot = max(
            p.custo_n * c.demanda_n(self.faixa_mo, ANTECESSORA_GRAMINEA)
            + p.custo_p2o5 * c.manutencao_p2o5
            + p.custo_k2o * c.manutencao_k2o
            + p.penal_balanco_n * c.saldo_n(self.faixa_mo) ** 2
            for c in self.culturas
        )
        if p.penal_alocacao_unica <= maior_custo_slot:
            avisos.append(
                f"penal_alocacao_unica ({p.penal_alocacao_unica:.0f}) não supera "
                f"o maior custo de alocação ({maior_custo_slot:.0f}): deixar um "
                "talhão vazio pode ficar mais barato que cultivá-lo."
            )

        maior_credito = max(
            (p.custo_n * c.credito_n_leguminosa(self.faixa_mo) for c in self.culturas),
            default=0.0,
        )
        if p.penal_repeticao_cultura <= maior_credito:
            avisos.append(
                f"penal_repeticao_cultura ({p.penal_repeticao_cultura:.0f}) é "
                f"menor que o maior crédito de N ({maior_credito:.0f}): repetir "
                "cultura pode compensar."
            )

        if p.penal_estacao <= maior_custo_slot:
            avisos.append(
                f"penal_estacao ({p.penal_estacao:.0f}) não supera o maior custo "
                f"de alocação ({maior_custo_slot:.0f}): cultivar fora de estação "
                "pode compensar."
            )

        # A repetição sazonal (t e t+2) é a única barreira contra monocultura
        # sazonal. Ela tem de superar a maior economia de N que repetir uma
        # cultura poderia comprar — isto é, o custo da dose mais alta do
        # catálogo, que seria evitada ao repetir a cultura de menor dose.
        maior_custo_n = max(
            p.custo_n * c.demanda_n(self.faixa_mo, ANTECESSORA_GRAMINEA)
            for c in self.culturas
        )
        if p.penal_repeticao_estacional <= maior_custo_n:
            avisos.append(
                f"penal_repeticao_estacional ({p.penal_repeticao_estacional:.0f}) "
                f"não supera o maior custo de N do catálogo "
                f"({maior_custo_n:.0f}): repetir a mesma cultura na mesma "
                "estação pode compensar, e o ótimo vira monocultura sazonal."
            )

        if exato:
            _, custo_viavel = self.otimo_viavel()
            try:
                _, custo_global = self.otimo_global()
            except ValueError:
                avisos.append(
                    "verificação exata ignorada: instância grande demais para "
                    "enumerar 2^n."
                )
                return avisos

            if custo_global < custo_viavel - 1e-9:
                avisos.append(
                    f"o mínimo global do QUBO ({custo_global:.2f}) é INFERIOR ao "
                    f"melhor plano viável ({custo_viavel:.2f}): as penalidades "
                    "não dominam o problema e o otimizador será recompensado "
                    "por violar as restrições. Aumente penal_alocacao_unica "
                    "e/ou penal_estacao."
                )

        return avisos

    def descricao(self) -> str:
        """Resumo das dimensões da instância."""
        return (
            f"Instância: {self.n_culturas} culturas x {self.n_talhoes} talhões "
            f"x {self.n_periodos} períodos\n"
            f"Variáveis binárias (qubits): {self.n_variaveis}\n"
            f"Espaço de busca irrestrito : 2^{self.n_variaveis} = "
            f"{2 ** self.n_variaveis:.3g}\n"
            f"Espaço viável (alocação única): "
            f"{self.n_culturas}^{self.n_talhoes * self.n_periodos} = "
            f"{self.n_culturas ** (self.n_talhoes * self.n_periodos):.3g}\n"
            f"Acoplamentos quadráticos não nulos: "
            f"{int(np.count_nonzero(np.triu(self.Q, k=1)))}"
        )


if __name__ == "__main__":
    from parametros import CENARIOS_POR_TAMANHO

    modelo = ModeloQUBO()
    print(modelo.descricao())

    avisos = modelo.verificar_penalidades()
    print("\nConsistência das penalidades:",
          "OK" if not avisos else "\n  - " + "\n  - ".join(avisos))

    print("\nMelhor plano viável do cenário-base, por enumeração exaustiva:\n")
    x_otimo, custo_otimo = modelo.otimo_viavel()
    print(modelo.relatorio(x_otimo))

    # Validação da formulação: o mínimo global do QUBO deve coincidir com o
    # melhor plano viável. Só é enumerável até ~24 variáveis.
    print("\n\nValidação da formulação (mínimo global x melhor plano viável)\n")
    for n_periodos in (2, 3, 4):
        pequeno = ModeloQUBO(n_periodos=n_periodos)
        _, c_viavel = pequeno.otimo_viavel()
        _, c_global = pequeno.otimo_global()
        estado = "OK" if abs(c_global - c_viavel) < 1e-9 else "PENALIDADES FRACAS"
        print(
            f"  {pequeno.n_variaveis:3d} vars (4 culturas x {n_periodos}p): "
            f"viável={c_viavel:9.2f}  global={c_global:9.2f}  [{estado}]"
        )

    # Escada de escalabilidade: o plano ótimo de cada degrau.
    print("\n\nEscada de escalabilidade (1 talhão x 4 períodos)\n")
    cab = (
        f"{'Cult':>4} {'Qubits':>7} {'Custo':>9} {'N(kg/ha)':>9} "
        f"{'SaldoN':>7} {'P2O5':>6} {'K2O':>6}  Plano ótimo"
    )
    print(cab)
    print("-" * 110)
    for n in sorted(CENARIOS_POR_TAMANHO):
        m = ModeloQUBO(culturas=CENARIOS_POR_TAMANHO[n])
        x, c = m.otimo_viavel()
        p2o5, k2o = m.adubacao_pk_total(x)
        plano = " > ".join(m.plano_de_rotacao(x)[0])
        print(
            f"{n:>4} {m.n_variaveis:>7} {c:>9.2f} {m.demanda_n_total(x):>9.0f} "
            f"{m.saldo_n_total(x):>+7.0f} {p2o5:>6.0f} {k2o:>6.0f}  {plano}"
        )
