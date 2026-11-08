"""
algoritmo_genetico.py
=====================

Algoritmo genético clássico aplicado ao QUBO de rotação de culturas.

Serve como método clássico de referência (*baseline*) na comparação com o QAOA.
Ambos otimizam a mesma matriz Q construída por :mod:`qubo_rotacao`; a única
diferença é a estratégia de busca.

Por que uma implementação própria em vez de uma biblioteca
----------------------------------------------------------
Bibliotecas como PyGAD ou DEAP são maduras, mas a comparação exigida pelo
trabalho depende de controlar com precisão o **orçamento de avaliações** da
função objetivo — a métrica que torna comparáveis dois paradigmas com custos
por iteração tão diferentes. Uma implementação direta em NumPy deixa esse
controle explícito e mantém a contagem auditável, além de eliminar uma
dependência externa. O laço evolutivo abaixo é o canônico descrito por
Goldberg (1989): seleção por torneio, cruzamento, mutação e elitismo.

A avaliação é vetorizada: uma geração inteira é avaliada com um único produto
matricial, o que torna o AG capaz de resolver instâncias muito maiores do que
as que o QAOA consegue simular.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qubo_rotacao import ModeloQUBO


@dataclass
class ResultadoAG:
    """Resultado de uma execução do algoritmo genético."""

    melhor_x: np.ndarray
    melhor_custo: float
    #: Melhor custo ao final de cada geração (curva de convergência)
    historico: list[float] = field(default_factory=list)
    n_avaliacoes: int = 0
    geracoes_executadas: int = 0
    #: Geração em que o melhor custo final foi atingido pela primeira vez
    geracao_convergencia: int = 0
    tempo_s: float = 0.0
    parou_por_estagnacao: bool = False

    def resumo(self) -> str:
        return (
            f"AG  | custo={self.melhor_custo:10.2f} | "
            f"gerações={self.geracoes_executadas:4d} | "
            f"avaliações={self.n_avaliacoes:7d} | "
            f"convergiu na geração {self.geracao_convergencia:4d} | "
            f"tempo={self.tempo_s:6.3f}s"
        )


@dataclass
class ConfiguracaoAG:
    """Hiperparâmetros do algoritmo genético.

    Os valores padrão foram escolhidos como ponto de partida convencional para
    problemas binários desta escala; a calibração fina deve ser reportada na
    seção de resultados do trabalho.
    """

    tamanho_populacao: int = 200
    n_geracoes: int = 300
    #: Probabilidade de inverter cada bit. ``None`` usa 1/n (regra usual).
    taxa_mutacao: float | None = None
    taxa_crossover: float = 0.9
    tamanho_torneio: int = 3
    #: Quantos melhores indivíduos passam intactos para a geração seguinte
    elitismo: int = 2
    #: Gerações sem melhora antes de encerrar. ``None`` desativa.
    paciencia: int | None = 60
    #: Se True, a população inicial já respeita a alocação única. Deixe em
    #: False para que o AG enfrente o mesmo espaço irrestrito {0,1}^n que o
    #: QAOA — condição para que a comparação seja justa.
    inicializacao_viavel: bool = False
    #: Se True, os três operadores passam a **preservar** a alocação única:
    #: população inicial viável, cruzamento por bloco e mutação por troca.
    #: É o espelho clássico do mixer XY do QAOA -- o AG passa a buscar
    #: somente entre os ``n_culturas ** n_periodos`` planos válidos, em vez
    #: de vagar pelos :math:`2^n` com penalidade. Rodar as duas variantes é o
    #: que separa, nos resultados, o efeito do *conhecimento da restrição* do
    #: efeito do método em si. Implica ``inicializacao_viavel``.
    preserva_viabilidade: bool = False


def _populacao_inicial(
    modelo: ModeloQUBO, cfg: ConfiguracaoAG, rng: np.random.Generator
) -> np.ndarray:
    """Gera a população inicial."""
    if not (cfg.inicializacao_viavel or cfg.preserva_viabilidade):
        return rng.integers(0, 2, size=(cfg.tamanho_populacao, modelo.n_variaveis)).astype(
            np.uint8
        )

    # Inicialização viável: sorteia exatamente uma cultura por talhão/período.
    X = np.zeros((cfg.tamanho_populacao, modelo.n_variaveis), dtype=np.uint8)
    for k in range(cfg.tamanho_populacao):
        for t in range(modelo.n_periodos):
            for j in range(modelo.n_talhoes):
                i = int(rng.integers(0, modelo.n_culturas))
                X[k, modelo.indice(i, j, t)] = 1
    return X


def _selecao_por_torneio(
    custos: np.ndarray, n_selecionados: int, tamanho_torneio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Seleção por torneio: devolve os índices dos vencedores.

    Sorteia ``tamanho_torneio`` competidores por vaga e escolhe o de menor
    custo. Pressão seletiva controlada pelo tamanho do torneio, sem exigir
    custos positivos (ao contrário da seleção por roleta) — relevante aqui,
    pois a função objetivo pode assumir valores negativos.
    """
    competidores = rng.integers(
        0, custos.size, size=(n_selecionados, tamanho_torneio)
    )
    custos_competidores = custos[competidores]
    vencedores = np.argmin(custos_competidores, axis=1)
    return competidores[np.arange(n_selecionados), vencedores]


def _crossover_uniforme(
    pais_a: np.ndarray, pais_b: np.ndarray, taxa: float, rng: np.random.Generator
) -> np.ndarray:
    """Cruzamento uniforme, gene a gene.

    Preferido ao cruzamento de ponto único porque a codificação x[i,j,t] é
    posicional: genes vizinhos no vetor pertencem ao mesmo talhão/período, e um
    corte de ponto único enviesaria a recombinação em favor de blocos
    arbitrários dessa indexação.
    """
    n_filhos, n_genes = pais_a.shape
    mascara = rng.random((n_filhos, n_genes)) < 0.5
    filhos = np.where(mascara, pais_a, pais_b)

    # Casais que não cruzam replicam o primeiro progenitor.
    sem_cruzamento = rng.random(n_filhos) >= taxa
    filhos[sem_cruzamento] = pais_a[sem_cruzamento]
    return filhos.astype(np.uint8)


def _blocos(modelo: ModeloQUBO) -> np.ndarray:
    """Índices das variáveis de cada bloco, em forma (n_blocos, n_culturas).

    Um **bloco** é um par (talhão, período): exatamente o conjunto de bits
    sobre o qual vale a restrição de alocação única. Toda a lógica que
    preserva a viabilidade trabalha em cima desta matriz.
    """
    return np.array(
        [[modelo.indice(i, j, t) for i in range(modelo.n_culturas)]
         for t in range(modelo.n_periodos)
         for j in range(modelo.n_talhoes)],
        dtype=np.intp,
    )


def _crossover_por_bloco(
    pais_a: np.ndarray, pais_b: np.ndarray, taxa: float,
    rng: np.random.Generator, blocos: np.ndarray,
) -> np.ndarray:
    """Cruzamento que herda blocos inteiros, preservando a alocação única.

    O cruzamento uniforme gene a gene sorteia cada bit de forma independente,
    e isso destrói a restrição: um bloco pode receber o bit 1 do progenitor A
    e também o bit 1 do progenitor B, ficando com duas culturas no mesmo
    período -- ou nenhum dos dois, ficando sem cultura. Aqui o sorteio é
    **por bloco**: o filho recebe o período inteiro de um dos progenitores.
    Como cada progenitor é viável, o filho também é.
    """
    n_filhos = pais_a.shape[0]
    escolhe_a = rng.random((n_filhos, blocos.shape[0])) < 0.5
    filhos = pais_b.copy()
    for b in range(blocos.shape[0]):
        linhas = np.nonzero(escolhe_a[:, b])[0]
        if linhas.size:
            recorte = np.ix_(linhas, blocos[b])
            filhos[recorte] = pais_a[recorte]

    # Casais que não cruzam replicam o primeiro progenitor.
    sem_cruzamento = rng.random(n_filhos) >= taxa
    filhos[sem_cruzamento] = pais_a[sem_cruzamento]
    return filhos.astype(np.uint8)


def _mutacao_por_troca(
    filhos: np.ndarray, blocos: np.ndarray, taxa_por_bloco: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Mutação que troca a cultura de um bloco, preservando a alocação única.

    A mutação por inversão de bit (``1 - bit``) muda a quantidade de 1s do
    bloco e portanto quebra a restrição -- é o análogo clássico exato do que
    o mixer :math:`\sum X_a` faz no QAOA. Aqui o bloco sorteado é reescrito
    com **exatamente uma** cultura, também sorteada: a contagem de 1s é
    preservada por construção, como no mixer XY.

    ``taxa_por_bloco`` é a probabilidade de cada bloco mutar. O padrão
    ``1/n_blocos`` faz mudar, em média, um bloco por indivíduo -- a mesma
    convenção do ``1/n`` usado na mutação por inversão de bit.
    """
    alvos = rng.random((filhos.shape[0], blocos.shape[0])) < taxa_por_bloco
    linhas, quais = np.nonzero(alvos)
    if linhas.size:
        sorteadas = rng.integers(0, blocos.shape[1], size=linhas.size)
        for k in range(linhas.size):
            idx = blocos[quais[k]]
            filhos[linhas[k], idx] = 0
            filhos[linhas[k], idx[sorteadas[k]]] = 1
    return filhos


def executar_ag(
    modelo: ModeloQUBO,
    cfg: ConfiguracaoAG | None = None,
    seed: int | None = None,
) -> ResultadoAG:
    """Executa o algoritmo genético sobre o QUBO informado.

    Parameters
    ----------
    modelo:
        Instância do problema (fornece a função objetivo).
    cfg:
        Hiperparâmetros; usa os padrões de :class:`ConfiguracaoAG` se omitido.
    seed:
        Semente do gerador aleatório. Informe sempre um valor nas execuções
        do experimento, para que os resultados sejam reprodutíveis.
    """
    cfg = cfg or ConfiguracaoAG()
    rng = np.random.default_rng(seed)

    # Quando os operadores preservam a viabilidade, a unidade de mutação
    # deixa de ser o bit e passa a ser o bloco (talhão, período).
    blocos = _blocos(modelo) if cfg.preserva_viabilidade else None
    if cfg.taxa_mutacao is not None:
        taxa_mutacao = cfg.taxa_mutacao
    elif blocos is not None:
        taxa_mutacao = 1.0 / blocos.shape[0]
    else:
        taxa_mutacao = 1.0 / modelo.n_variaveis

    t_inicio = time.perf_counter()

    populacao = _populacao_inicial(modelo, cfg, rng)
    custos = modelo.avaliar_populacao(populacao)
    n_avaliacoes = populacao.shape[0]

    k_melhor = int(np.argmin(custos))
    melhor_x = populacao[k_melhor].copy()
    melhor_custo = float(custos[k_melhor])
    geracao_convergencia = 0

    historico: list[float] = [melhor_custo]
    geracoes_executadas = 0
    parou_por_estagnacao = False
    geracoes_sem_melhora = 0

    for geracao in range(1, cfg.n_geracoes + 1):
        # --- elitismo: preserva os melhores sem alteração -----------------
        if cfg.elitismo > 0:
            elite_idx = np.argpartition(custos, cfg.elitismo - 1)[: cfg.elitismo]
            elite = populacao[elite_idx].copy()
        else:
            elite = np.empty((0, modelo.n_variaveis), dtype=np.uint8)

        # --- seleção ------------------------------------------------------
        n_filhos = cfg.tamanho_populacao - elite.shape[0]
        idx_a = _selecao_por_torneio(custos, n_filhos, cfg.tamanho_torneio, rng)
        idx_b = _selecao_por_torneio(custos, n_filhos, cfg.tamanho_torneio, rng)

        # --- cruzamento e mutação -----------------------------------------
        if blocos is not None:
            filhos = _crossover_por_bloco(
                populacao[idx_a], populacao[idx_b], cfg.taxa_crossover,
                rng, blocos,
            )
            filhos = _mutacao_por_troca(filhos, blocos, taxa_mutacao, rng)
        else:
            filhos = _crossover_uniforme(
                populacao[idx_a], populacao[idx_b], cfg.taxa_crossover, rng
            )
            mutacoes = rng.random(filhos.shape) < taxa_mutacao
            filhos = np.where(mutacoes, 1 - filhos, filhos).astype(np.uint8)

        populacao = np.vstack([elite, filhos])
        custos = modelo.avaliar_populacao(populacao)
        n_avaliacoes += filhos.shape[0]
        geracoes_executadas = geracao

        k_melhor = int(np.argmin(custos))
        if custos[k_melhor] < melhor_custo - 1e-12:
            melhor_custo = float(custos[k_melhor])
            melhor_x = populacao[k_melhor].copy()
            geracao_convergencia = geracao
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1

        historico.append(melhor_custo)

        if cfg.paciencia is not None and geracoes_sem_melhora >= cfg.paciencia:
            parou_por_estagnacao = True
            break

    return ResultadoAG(
        melhor_x=melhor_x.astype(int),
        melhor_custo=melhor_custo,
        historico=historico,
        n_avaliacoes=n_avaliacoes,
        geracoes_executadas=geracoes_executadas,
        geracao_convergencia=geracao_convergencia,
        tempo_s=time.perf_counter() - t_inicio,
        parou_por_estagnacao=parou_por_estagnacao,
    )


def executar_ag_repetido(
    modelo: ModeloQUBO,
    n_execucoes: int = 30,
    cfg: ConfiguracaoAG | None = None,
    seed_base: int = 0,
) -> list[ResultadoAG]:
    """Executa o AG várias vezes com sementes distintas.

    Metaheurísticas são estocásticas: uma única execução não caracteriza o
    desempenho do método. As métricas do trabalho (média, desvio padrão, taxa
    de acerto do ótimo) devem sempre vir de múltiplas execuções independentes.
    """
    return [
        executar_ag(modelo, cfg=cfg, seed=seed_base + k) for k in range(n_execucoes)
    ]


if __name__ == "__main__":
    modelo = ModeloQUBO()
    print(modelo.descricao())

    print("\nMelhor plano viável (enumeração exaustiva):")
    x_otimo, custo_otimo = modelo.otimo_viavel()
    print(f"  custo = {custo_otimo:.2f}")

    print("\nExecutando o algoritmo genético (10 execuções independentes)...\n")
    resultados = executar_ag_repetido(modelo, n_execucoes=10, seed_base=42)

    for k, r in enumerate(resultados):
        marca = "  <- melhor plano" if abs(r.melhor_custo - custo_otimo) < 1e-9 else ""
        print(f"  [{k:2d}] {r.resumo()}{marca}")

    custos = np.array([r.melhor_custo for r in resultados])
    tempos = np.array([r.tempo_s for r in resultados])
    acertos = int(np.sum(np.abs(custos - custo_otimo) < 1e-9))

    print(
        f"\nCusto  : média={custos.mean():.2f}  desvio={custos.std(ddof=1):.2f}  "
        f"melhor={custos.min():.2f}"
    )
    print(f"Tempo  : média={tempos.mean():.3f}s  desvio={tempos.std(ddof=1):.3f}s")
    print(f"Execuções que atingiram o melhor plano: {acertos}/{len(resultados)}")

    print("\nMelhor plano encontrado:\n")
    melhor = min(resultados, key=lambda r: r.melhor_custo)
    print(modelo.relatorio(melhor.melhor_x))
