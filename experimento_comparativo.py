"""
experimento_comparativo.py
==========================

Executa a comparação sistemática entre o QAOA e o algoritmo genético sobre o
mesmo QUBO, em instâncias de tamanho crescente, e grava os resultados em CSV
para alimentar a seção *Resultados e Discussão* do trabalho.

Métricas coletadas
------------------
Correspondem às definidas na fundamentação teórica do TCC:

- **Qualidade da solução** — custo obtido, desvio percentual em relação ao
  ótimo global (conhecido por enumeração exaustiva, e não apenas o melhor
  entre os dois métodos) e viabilidade da alocação.
- **Eficiência computacional** — tempo de execução, número de avaliações da
  função objetivo, gerações até a convergência (AG) e iterações do otimizador
  clássico (QAOA).
- **Escalabilidade** — todas as métricas em função do número de variáveis.
- **Recursos computacionais** — qubits, profundidade e contagem de portas do
  circuito (QAOA).
- **Grandezas agronômicas** — demanda e saldo de nitrogênio, adubação de P e K
  do plano. São estas, e não o valor da função objetivo, que têm unidade
  física e significado para a discussão.

Desenho do experimento
----------------------
O cenário é fixo em **1 talhão x 4 períodos** (verão/inverno ao longo de dois
anos agrícolas) e a escala cresce pelo **número de culturas candidatas**, de 4
a 8 — isto é, de 16 a 32 variáveis binárias.

Uso
---
::

    python experimento_comparativo.py                    # configuração padrão
    python experimento_comparativo.py --execucoes 30     # mais repetições
    python experimento_comparativo.py --max-qubits-qaoa 24
    python experimento_comparativo.py --precisao single  # metade da memória
    python experimento_comparativo.py --sem-qaoa         # apenas o AG

Advertência sobre o tempo e a memória
-------------------------------------
O custo da simulação do QAOA dobra a cada qubit acrescentado, e a memória do
vetor de estado também: 24 qubits exigem 0,25 GB, 28 exigem 4 GB e 32 exigem
64 GB em precisão dupla (a metade disso em precisão simples). O padrão
``--max-qubits-qaoa 28`` corresponde ao maior degrau simulável numa máquina
pessoal; o degrau de 8 culturas fica fora do alcance do QAOA e é executado
apenas pelo AG.

O AG, por não simular nada, permanece na casa dos milissegundos em qualquer
das instâncias previstas.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from algoritmo_genetico import ConfiguracaoAG, executar_ag
from parametros import CENARIOS_POR_TAMANHO, validar_cenario
from progresso import ContadorDeAvaliacoes, Progresso, formatar_duracao
from qaoa_rotacao import QISKIT_DISPONIVEL, estimar_memoria_gb, executar_qaoa
from qubo_rotacao import ModeloQUBO

#: Cenário fixo do estudo: um talhão, quatro períodos (verão/inverno x 2 anos).
N_TALHOES = 1
N_PERIODOS = 4

#: Escada de escalabilidade: o problema cresce pelo **número de culturas**.
#:
#: Com ``n`` culturas o modelo tem ``4n`` variáveis binárias, então a escada
#: 4..8 cobre de 16 a 32 qubits. Os conjuntos vêm de
#: :data:`CENARIOS_POR_TAMANHO`, que alterna a estação enriquecida a cada
#: degrau para que o espaço de decisão cresça de forma equilibrada.
#:
#: Crescer por talhões, como se fazia antes, **não** ampliava o problema:
#: nenhum termo do modelo liga um talhão a outro, então um cenário de dois
#: talhões é literalmente dois problemas independentes, e seu ótimo é o dobro
#: do de um talhão.
INSTANCIAS: list[int] = [4, 5, 6, 7, 8]

CAMPOS_CSV = [
    "instancia",
    "n_culturas",
    "n_talhoes",
    "n_periodos",
    "n_variaveis",
    "metodo",
    "execucao",
    "instante",
    "custo",
    "custo_otimo",
    "desvio_percentual",
    "atingiu_otimo",
    "alocacao_valida",
    "demanda_n_kg_ha",
    "saldo_n_kg_ha",
    "adubacao_p2o5_kg_ha",
    "adubacao_k2o_kg_ha",
    "tempo_s",
    "n_avaliacoes",
    "geracoes",
    "geracao_convergencia",
    "qaoa_p",
    "valor_esperado",
    "profundidade_circuito",
    "n_portas",
    "prob_melhor",
    "memoria_estimada_gb",
]


class EscritorIncremental:
    """Grava cada execucao no CSV na hora em que ela termina, e sabe retomar.

    Por que nao acumular tudo em memoria e gravar no fim
    ----------------------------------------------------
    O experimento leva horas. Gravar so no fim significa perder tudo se a
    maquina for desligada, se faltar memoria na instancia grande ou se o
    processo for interrompido. Aqui cada linha vai para o disco e e descarregada
    na hora.

    E, mais importante para o cronograma do trabalho: como o escritor sabe
    quais combinacoes (instancia, metodo, execucao) ja existem no arquivo, uma
    segunda chamada **pula o que esta feito e continua de onde parou**. Isso
    torna o numero de repeticoes um parametro que pode crescer depois: ir de 5
    para 30 execucoes custa rodar 25, nao refazer 30.
    """

    def __init__(self, caminho: Path, retomar: bool = True) -> None:
        self.caminho = caminho
        self.feitos: set[tuple[str, str, str]] = set()
        caminho.parent.mkdir(parents=True, exist_ok=True)

        existe = caminho.exists() and caminho.stat().st_size > 0
        if existe and retomar:
            with caminho.open(encoding="utf-8", newline="") as f:
                leitor = csv.DictReader(f)
                # A retomada grava em modo "a", com os nomes de coluna atuais.
                # Se o arquivo veio de uma versao do script com outras colunas,
                # as linhas novas sairiam desalinhadas do cabecalho antigo: um
                # CSV corrompido em silencio, o pior desfecho possivel depois
                # de uma noite de execucao. Melhor parar e exigir uma decisao.
                if (leitor.fieldnames is not None
                        and list(leitor.fieldnames) != CAMPOS_CSV):
                    novas = sorted(set(CAMPOS_CSV) - set(leitor.fieldnames))
                    raise SystemExit(
                        f"{caminho} tem o cabecalho de outra versao do "
                        f"script (colunas novas: {novas or 'nenhuma'}). "
                        f"Arquive o arquivo ou rode com --recomecar."
                    )
                for linha in leitor:
                    self.feitos.add((
                        str(linha.get("instancia", "")),
                        str(linha.get("metodo", "")),
                        str(linha.get("execucao", "")),
                    ))

        modo = "a" if (existe and retomar) else "w"
        self._f = caminho.open(modo, newline="", encoding="utf-8")
        self._escritor = csv.DictWriter(
            self._f, fieldnames=CAMPOS_CSV, extrasaction="ignore"
        )
        if modo == "w":
            self._escritor.writeheader()
            self._f.flush()

    def ja_feito(self, instancia: str, metodo: str, execucao: int) -> bool:
        return (str(instancia), str(metodo), str(execucao)) in self.feitos

    def escrever(self, linha: dict) -> None:
        self._escritor.writerow(linha)
        self._f.flush()
        self.feitos.add((
            str(linha.get("instancia", "")),
            str(linha.get("metodo", "")),
            str(linha.get("execucao", "")),
        ))

    def fechar(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "EscritorIncremental":
        return self

    def __exit__(self, *_exc) -> None:
        self.fechar()


@dataclass
class Configuracao:
    execucoes: int = 10
    seed_base: int = 42
    qaoa_p: int = 2
    qaoa_maxiter: int = 150
    qaoa_shots: int = 4096
    #: Reinícios do otimizador. Vários pontos de partida e o melhor entre eles
    #: foi o que tornou a qualidade do QAOA reprodutível: com um único ponto a
    #: qualidade oscilava sem padrão com p.
    qaoa_n_reinicios: int = 3
    #: Instâncias maiores não são simuladas. O teto prático é a memória do
    #: vetor de estado, que dobra a cada qubit: 28 qubits exigem 4 GB em
    #: precisão dupla e 2 GB em simples; 32 qubits exigem 64 e 32 GB.
    max_qubits_qaoa: int = 28
    #: "double" ou "single"; precisão simples corta a memória pela metade.
    precisao: str = "double"
    #: Threads do Aer. Oito, não todas as 32 lógicas: medido em 24 qubits, o
    #: tempo cai de 11,03 s (1 thread) para 3,14 s (8) e volta a subir para
    #: 3,65 s usando todas, porque a simulação é limitada pela banda de
    #: memória e não pelo processamento. Zero devolve o padrão do Aer.
    threads: int = 8
    executar_qaoa: bool = True
    saida: Path = Path("resultados/comparativo.csv")
    #: Pula as combinacoes que ja constam do CSV, em vez de recomeca-lo.
    retomar: bool = True
    #: Mixers do QAOA a executar, em ordem. "X" é o mixer padrão da
    #: literatura, que ignora a restrição de alocação única; "XY" preserva a
    #: restrição e confina a busca ao subespaço viável. Rodar os dois é o que
    #: separa o efeito do *conhecimento da restrição* do efeito do método.
    mixers: tuple[str, ...] = ("X",)
    #: Variantes do AG a executar. ``"padrao"`` usa cruzamento uniforme e
    #: mutação por inversão de bit, que ignoram a alocação única; ``"viavel"``
    #: usa cruzamento por bloco e mutação por troca, que a preservam. Junto
    #: com ``mixers``, forma o desenho 2x2 que separa o efeito do
    #: conhecimento da restrição do efeito do método.
    ags: tuple[str, ...] = ("padrao",)
    #: Log de acompanhamento, uma linha por execucao concluida.
    log: Path = Path("resultados/execucao.log")


def _linha_base(modelo: ModeloQUBO, rotulo: str, custo_otimo: float | None,
                precisao: str) -> dict:
    """Campos comuns a AG e QAOA, inclusive o instante de conclusão.

    ``instante`` é gravado quando a execução **termina**: a função é chamada
    com o resultado em mãos. O início, se for preciso, é ``instante`` menos
    ``tempo_s``. Datar cada linha é o que permite reunir num mesmo CSV
    execuções feitas em noites diferentes e ainda saber o que rodou quando --
    indispensável porque o experimento é retomável por construção.
    """
    return {
        "instancia": rotulo,
        "instante": datetime.now().isoformat(timespec="seconds"),
        "n_culturas": modelo.n_culturas,
        "n_talhoes": modelo.n_talhoes,
        "n_periodos": modelo.n_periodos,
        "n_variaveis": modelo.n_variaveis,
        "custo_otimo": "" if custo_otimo is None else f"{custo_otimo:.4f}",
        "memoria_estimada_gb": (
            f"{estimar_memoria_gb(modelo.n_variaveis, precisao):.4f}"
        ),
        "geracoes": "",
        "geracao_convergencia": "",
        "qaoa_p": "",
        "valor_esperado": "",
        "profundidade_circuito": "",
        "n_portas": "",
        "prob_melhor": "",
    }


def _metricas_agronomicas(modelo: ModeloQUBO, x) -> dict:
    """Grandezas com significado agronômico, independentes dos pesos.

    O ``custo`` da função objetivo é um escalar com pesos e penalidades
    embutidos, sem unidade física. São estes os números que se defendem na
    discussão dos resultados.
    """
    p2o5, k2o = modelo.adubacao_pk_total(x)
    return {
        "demanda_n_kg_ha": f"{modelo.demanda_n_total(x):.1f}",
        "saldo_n_kg_ha": f"{modelo.saldo_n_total(x):.1f}",
        "adubacao_p2o5_kg_ha": f"{p2o5:.1f}",
        "adubacao_k2o_kg_ha": f"{k2o:.1f}",
    }


def _desvio(custo: float, custo_otimo: float | None) -> str:
    if custo_otimo is None or abs(custo_otimo) < 1e-12:
        return ""
    return f"{100.0 * (custo - custo_otimo) / abs(custo_otimo):.4f}"


def _rotulo_ag(variante: str) -> str:
    """Nome da variante do AG no CSV.

    ``"padrao"`` conserva o rótulo ``AG`` puro para que as execuções já
    gravadas sigam valendo na retomada; ``"viavel"`` é a variante cujos
    operadores preservam a alocação única -- o espelho clássico do mixer XY.
    """
    return "AG" if variante == "padrao" else "AG-VIAVEL"


def _rotulo_metodo(mixer: str) -> str:
    """Nome do método no CSV.

    O mixer X conserva o rótulo ``QAOA`` puro, sem sufixo, de propósito: as
    execuções já gravadas usam esse nome, e manter o rótulo é o que permite
    que a retomada as reconheça em vez de repeti-las.
    """
    return "QAOA" if mixer.upper() == "X" else f"QAOA-{mixer.upper()}"


def _plano_de_execucao(cfg: Configuracao) -> list[tuple[int, str, int]]:
    """Lista todas as (n_culturas, metodo, execucao) que o experimento fará.

    Serve para saber o total **antes** de começar, que é o que permite mostrar
    "15/100" e prever o horário de término em vez de só contar para cima.
    """
    plano: list[tuple[int, str, int]] = []
    for n_culturas in INSTANCIAS:
        n_vars = n_culturas * N_TALHOES * N_PERIODOS
        for variante in cfg.ags:
            for k in range(cfg.execucoes):
                plano.append((n_culturas, _rotulo_ag(variante), k))
        if cfg.executar_qaoa and QISKIT_DISPONIVEL and n_vars <= cfg.max_qubits_qaoa:
            for mixer in cfg.mixers:
                for k in range(cfg.execucoes):
                    plano.append((n_culturas, _rotulo_metodo(mixer), k))
    return plano


#: Segundos de UMA avaliação da função de custo do QAOA, por número de qubits,
#: medidos nesta máquina com 8 threads (``resultados/escalabilidade.csv`` e a
#: medição direta em 28 qubits). **Não entram em nenhum resultado do
#: trabalho**: servem apenas para ponderar a previsão de término da barra.
SEGUNDOS_POR_AVALIACAO: dict[int, float] = {
    16: 0.04,
    20: 0.14,
    24: 5.04,
    28: 68.5,
    32: 1100.0,
}


def _pesos_do_plano(plano: list[tuple[int, str, int]],
                    cfg: Configuracao) -> list[float]:
    """Custo esperado de cada etapa do plano, em segundos.

    A barra prevê o término por regra de três. Sem pesos ela supõe etapas de
    custo igual, e as etapas aqui diferem por cinco ordens de grandeza: uma
    execução do AG leva centésimos de segundo, uma do QAOA em 24 qubits leva
    cerca de 15 minutos. A previsão diria "faltam 2s" com duas horas pela
    frente. Os valores não precisam estar certos em absoluto -- basta que
    guardem entre si a proporção correta.
    """
    pesos: list[float] = []
    for n_culturas, metodo, _k in plano:
        n_vars = n_culturas * N_TALHOES * N_PERIODOS
        if metodo.startswith("AG"):
            pesos.append(0.05)
            continue
        if metodo.startswith("QAOA"):
            por_aval = SEGUNDOS_POR_AVALIACAO.get(n_vars, 2.5e-7 * 2 ** n_vars)
            # O mixer XY usa portas de dois qubits e um estado inicial
            # preparado: cerca do dobro das portas do mixer X, medido em
            # 16 qubits (1089 contra 537, com p=2).
            if metodo == "QAOA-XY":
                por_aval *= 2.0
            pesos.append(cfg.qaoa_maxiter * cfg.qaoa_n_reinicios * por_aval)
        else:
            pesos.append(0.05)
    return pesos


def executar_experimento(cfg: Configuracao) -> list[dict]:
    """Roda todas as instâncias e devolve as linhas de resultado."""
    linhas: list[dict] = []

    plano = _plano_de_execucao(cfg)
    escritor = EscritorIncremental(cfg.saida, retomar=cfg.retomar)
    pendentes = [t for t in plano
                 if not escritor.ja_feito(f"{t[0]} culturas", t[1], t[2])]
    if escritor.feitos:
        print(f"Retomando: {len(plano) - len(pendentes)} execuções já no CSV, "
              f"{len(pendentes)} a fazer.")

    barra = Progresso(
        total=len(pendentes), rotulo="experimento", arquivo_log=cfg.log,
        pesos=_pesos_do_plano(pendentes, cfg),
    )
    barra.registrar(
        f"plano: {len(plano)} execuções, {len(pendentes)} pendentes | "
        f"instâncias={INSTANCIAS} execucoes={cfg.execucoes} "
        f"p={cfg.qaoa_p} maxiter={cfg.qaoa_maxiter} precisao={cfg.precisao}"
    )
    barra.desenhar(forcar=True)

    for n_culturas in INSTANCIAS:
        culturas = CENARIOS_POR_TAMANHO[n_culturas]
        modelo = ModeloQUBO(
            culturas=culturas, n_talhoes=N_TALHOES, n_periodos=N_PERIODOS
        )
        rotulo = f"{n_culturas} culturas"

        print(f"\n{'=' * 72}")
        print(f"Instância {rotulo}  ->  {modelo.n_variaveis} variáveis binárias")
        print("=" * 72)

        for problema in validar_cenario(culturas):
            print(f"  [cenário mal formulado] {problema}")

        # A checagem exata compara o mínimo global do QUBO com o melhor plano
        # viável; só é possível nas instâncias pequenas, mas é justamente nelas
        # que se valida a formulação antes de escalar.
        exato = modelo.n_variaveis <= 20
        for aviso in modelo.verificar_penalidades(exato=exato):
            print(f"  [aviso] {aviso}")

        # --- melhor plano viável, usado como referência --------------------
        custo_otimo: float | None = None
        n_slots = modelo.n_talhoes * modelo.n_periodos
        if modelo.n_culturas**n_slots <= 20_000_000:
            t0 = time.perf_counter()
            x_otimo, custo_otimo = modelo.otimo_viavel()
            print(
                f"  Melhor plano viável (enumeração): {custo_otimo:.2f}  "
                f"[{time.perf_counter() - t0:.2f}s]"
            )
            print(f"    {' > '.join(modelo.plano_de_rotacao(x_otimo)[0])}")
        else:
            print("  Referência: espaço viável grande demais para enumerar.")

        # --- algoritmo genético -------------------------------------------
        for variante in cfg.ags:
            metodo = _rotulo_ag(variante)
            print(f"  {metodo}: {cfg.execucoes} execuções independentes...",
                  flush=True)
            custos_ag: list[float] = []
            for k in range(cfg.execucoes):
                if escritor.ja_feito(rotulo, metodo, k):
                    continue
                barra.atualizar(f"{modelo.n_variaveis}q {metodo} exec {k + 1}/"
                                f"{cfg.execucoes}")
                r = executar_ag(
                    modelo,
                    cfg=ConfiguracaoAG(
                        preserva_viabilidade=(variante == "viavel")
                    ),
                    seed=cfg.seed_base + k,
                )
                custos_ag.append(r.melhor_custo)
                linha = _linha_base(modelo, rotulo, custo_otimo, cfg.precisao)
                linha.update(
                    metodo=metodo,
                    execucao=k,
                    custo=f"{r.melhor_custo:.4f}",
                    desvio_percentual=_desvio(r.melhor_custo, custo_otimo),
                    atingiu_otimo=(
                        ""
                        if custo_otimo is None
                        else int(abs(r.melhor_custo - custo_otimo) < 1e-9)
                    ),
                    alocacao_valida=int(modelo.alocacao_valida(r.melhor_x)),
                    tempo_s=f"{r.tempo_s:.6f}",
                    n_avaliacoes=r.n_avaliacoes,
                    geracoes=r.geracoes_executadas,
                    geracao_convergencia=r.geracao_convergencia,
                    **_metricas_agronomicas(modelo, r.melhor_x),
                )
                linhas.append(linha)
                escritor.escrever(linha)
                barra.passo(f"{modelo.n_variaveis}q {metodo} exec {k + 1} "
                            f"custo={r.melhor_custo:.2f}")

            if custos_ag:
                arr = np.array(custos_ag)
                desvio = arr.std(ddof=1) if arr.size > 1 else 0.0
                print(
                    f"    {metodo}: custo médio={arr.mean():.2f} (±{desvio:.2f}), "
                    f"melhor={arr.min():.2f}"
                )

        # --- QAOA ----------------------------------------------------------
        if not cfg.executar_qaoa:
            continue
        if not QISKIT_DISPONIVEL:
            print("  QAOA: Qiskit não instalado — etapa ignorada.")
            continue
        if modelo.n_variaveis > cfg.max_qubits_qaoa:
            print(
                f"  QAOA: {modelo.n_variaveis} qubits acima do limite de "
                f"{cfg.max_qubits_qaoa} (~"
                f"{estimar_memoria_gb(modelo.n_variaveis, cfg.precisao):.1f} GB"
                f" em precisão {cfg.precisao}) — ignorado."
            )
            continue

        for mixer in cfg.mixers:
            metodo = _rotulo_metodo(mixer)
            print(
                f"  {metodo} (p={cfg.qaoa_p}, mixer {mixer}): "
                f"{cfg.execucoes} execuções...",
                flush=True,
            )
            custos_qaoa: list[float] = []
            for k in range(cfg.execucoes):
                if escritor.ja_feito(rotulo, metodo, k):
                    continue
                # Numa instância de 28 qubits uma única avaliação passa de um
                # minuto, e uma execução inteira leva horas. Sem este contador a
                # barra ficaria imóvel todo esse tempo, indistinguível de um
                # travamento.
                contador = ContadorDeAvaliacoes(
                    barra,
                    total_previsto=cfg.qaoa_maxiter * max(1, cfg.qaoa_n_reinicios),
                    prefixo=f"{modelo.n_variaveis}q {metodo} exec {k + 1}/"
                            f"{cfg.execucoes} ",
                )
                r = executar_qaoa(
                    modelo,
                    p=cfg.qaoa_p,
                    maxiter=cfg.qaoa_maxiter,
                    shots=cfg.qaoa_shots,
                    seed=cfg.seed_base + k,
                    memoria_maxima_gb=(
                        estimar_memoria_gb(cfg.max_qubits_qaoa, cfg.precisao) + 1.0
                    ),
                    precisao=cfg.precisao,
                    n_reinicios=cfg.qaoa_n_reinicios,
                    callback=contador,
                    threads=cfg.threads,
                    mixer=mixer,
                )
                custos_qaoa.append(r.melhor_custo)
                linha = _linha_base(modelo, rotulo, custo_otimo, cfg.precisao)
                linha.update(
                    metodo=metodo,
                    execucao=k,
                    custo=f"{r.melhor_custo:.4f}",
                    desvio_percentual=_desvio(r.melhor_custo, custo_otimo),
                    atingiu_otimo=(
                        ""
                        if custo_otimo is None
                        else int(abs(r.melhor_custo - custo_otimo) < 1e-9)
                    ),
                    alocacao_valida=int(modelo.alocacao_valida(r.melhor_x)),
                    tempo_s=f"{r.tempo_s:.6f}",
                    n_avaliacoes=r.n_avaliacoes,
                    qaoa_p=r.p,
                    valor_esperado=f"{r.valor_esperado:.4f}",
                    profundidade_circuito=r.profundidade_circuito,
                    n_portas=r.n_portas,
                    prob_melhor=f"{r.prob_melhor:.6f}",
                    **_metricas_agronomicas(modelo, r.melhor_x),
                )
                linhas.append(linha)
                escritor.escrever(linha)
                barra.passo(f"{modelo.n_variaveis}q {metodo} exec {k + 1} "
                            f"custo={r.melhor_custo:.2f} "
                            f"({formatar_duracao(r.tempo_s)})")

            if custos_qaoa:
                arr = np.array(custos_qaoa)
                desvio = arr.std(ddof=1) if arr.size > 1 else 0.0
                print(
                    f"    {metodo}: custo médio={arr.mean():.2f} (±{desvio:.2f}), "
                    f"melhor={arr.min():.2f}"
                )

    barra.encerrar(f"CSV em {cfg.saida}")
    escritor.fechar()
    return linhas


def gravar_csv(linhas: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV, extrasaction="ignore")
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow(linha)


def imprimir_sumario(linhas: list[dict]) -> None:
    """Tabela consolidada por instância e método."""
    print(f"\n\n{'=' * 96}")
    print("SUMÁRIO COMPARATIVO")
    print("=" * 96)
    cabecalho = (
        f"{'Instância':<14} {'Vars':>5} {'Método':<6} {'Custo médio':>12} "
        f"{'Desvio %':>10} {'Ótimo':>7} {'Viável':>7} {'Tempo (s)':>11}"
    )
    print(cabecalho)
    print("-" * len(cabecalho))

    vistos: list[tuple[str, str]] = []
    for linha in linhas:
        chave = (linha["instancia"], linha["metodo"])
        if chave in vistos:
            continue
        vistos.append(chave)

        grupo = [
            r
            for r in linhas
            if (r["instancia"], r["metodo"]) == chave
        ]
        custos = np.array([float(r["custo"]) for r in grupo])
        tempos = np.array([float(r["tempo_s"]) for r in grupo])
        desvios = [r["desvio_percentual"] for r in grupo if r["desvio_percentual"]]
        otimos = [r["atingiu_otimo"] for r in grupo if r["atingiu_otimo"] != ""]
        viaveis = [int(r["alocacao_valida"]) for r in grupo]

        desvio_txt = (
            f"{np.mean([float(d) for d in desvios]):>10.2f}" if desvios else f"{'-':>10}"
        )
        otimo_txt = (
            f"{sum(int(o) for o in otimos)}/{len(otimos)}" if otimos else "-"
        )

        print(
            f"{linha['instancia']:<14} {linha['n_variaveis']:>5} "
            f"{linha['metodo']:<6} {custos.mean():>12.2f} {desvio_txt} "
            f"{otimo_txt:>7} {f'{sum(viaveis)}/{len(viaveis)}':>7} "
            f"{tempos.mean():>11.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparação entre QAOA e algoritmo genético no QUBO de "
        "rotação de culturas."
    )
    parser.add_argument("--execucoes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--qaoa-p", type=int, default=2)
    parser.add_argument("--qaoa-maxiter", type=int, default=150)
    parser.add_argument("--qaoa-shots", type=int, default=4096)
    parser.add_argument(
        "--max-qubits-qaoa",
        type=int,
        default=28,
        help="Instâncias maiores que isto não são simuladas (memória). "
        "28 qubits = 7 culturas = 4 GB; 32 qubits = 8 culturas = 64 GB.",
    )
    parser.add_argument(
        "--precisao",
        choices=("double", "single"),
        default="double",
        help="Precisão do vetor de estado; 'single' corta a memória pela metade.",
    )
    parser.add_argument("--qaoa-reinicios", type=int, default=3,
                        help="Pontos de partida do otimizador por execução.")
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Threads do Aer. O padrão 8 foi medido como o melhor em 24 "
             "qubits; usar todas as 32 é 16%% mais lento. 0 = padrão do Aer.",
    )
    parser.add_argument("--sem-qaoa", action="store_true")
    parser.add_argument(
        "--ags", nargs="+", choices=("padrao", "viavel"), default=["padrao"],
        help="Variantes do AG a executar. 'viavel' preserva a alocação "
             "única nos três operadores. Exemplo: --ags padrao viavel",
    )
    parser.add_argument(
        "--mixers", nargs="+", choices=("X", "XY"), default=["X"],
        help="Mixers do QAOA a executar. X é o padrão da literatura; XY "
             "preserva a alocação única. Exemplo: --mixers X XY",
    )
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparativo.csv"))
    parser.add_argument("--log", type=Path, default=Path("resultados/execucao.log"))
    parser.add_argument(
        "--recomecar",
        action="store_true",
        help="Descarta o CSV existente em vez de continuar de onde parou.",
    )
    args = parser.parse_args()

    cfg = Configuracao(
        execucoes=args.execucoes,
        seed_base=args.seed_base,
        qaoa_p=args.qaoa_p,
        qaoa_maxiter=args.qaoa_maxiter,
        qaoa_shots=args.qaoa_shots,
        qaoa_n_reinicios=args.qaoa_reinicios,
        max_qubits_qaoa=args.max_qubits_qaoa,
        precisao=args.precisao,
        threads=args.threads,
        mixers=tuple(args.mixers),
        ags=tuple(args.ags),
        executar_qaoa=not args.sem_qaoa,
        saida=args.saida,
        retomar=not args.recomecar,
        log=args.log,
    )

    t0 = time.perf_counter()
    linhas = executar_experimento(cfg)

    # O sumário lê o CSV completo, não só as linhas desta sessão: numa
    # retomada as execuções anteriores também contam para as médias.
    todas = linhas
    if cfg.saida.exists():
        with cfg.saida.open(encoding="utf-8", newline="") as f:
            todas = list(csv.DictReader(f))
    imprimir_sumario(todas)

    # Nada de gravar_csv() aqui: o CSV já foi escrito linha por linha pelo
    # EscritorIncremental. Reescrevê-lo apagaria as execuções retomadas.
    print(f"\n{len(linhas)} execuções nesta sessão, "
          f"{len(todas)} no total em {cfg.saida}")
    print(f"Log de acompanhamento: {cfg.log}")
    print(f"Tempo total: {formatar_duracao(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
