"""
parametros.py
=============

Parâmetros agronômicos das culturas candidatas ao cenário de rotação de culturas
da região da AMESC (Extremo Sul Catarinense), em Cambissolos.

TODOS os valores numéricos deste módulo foram extraídos do:

    Manual de Calagem e Adubação para os Estados do Rio Grande do Sul e de
    Santa Catarina. Comissão de Química e Fertilidade do Solo - RS/SC
    (CQFS-RS/SC), 11a edição, 2016. (chave BibTeX: ManualCQFS)

Cada campo indica, em comentário, a tabela/seção e a página de origem, de modo
que qualquer valor possa ser reconferido no PDF original. Nenhum valor foi
estimado ou arbitrado: o que não consta no Manual está explicitamente marcado
como premissa do modelo.

Convenção de unidades
---------------------
- Rendimento de referência ............ t/ha
- Teor de nutriente no grão ........... kg de nutriente / t de grãos
- Adubação de manutenção .............. kg de nutriente / ha
- Demanda de N ........................ kg de N / ha

Observação: o Manual expressa fósforo e potássio como óxidos (P2O5 e K2O), e
esta é a convenção mantida aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Faixas de teor de matéria orgânica (MO) do solo usadas pelo Manual
# ---------------------------------------------------------------------------
# O Manual estratifica a recomendação de nitrogênio em três faixas de MO.
# Cambissolos da AMESC sob manejo conservacionista situam-se tipicamente na
# faixa intermediária, adotada como padrão do modelo.
FAIXA_MO_BAIXA = "<=2.5"
FAIXA_MO_MEDIA = "2.6-5.0"
FAIXA_MO_ALTA = ">5.0"

FAIXAS_MO = (FAIXA_MO_BAIXA, FAIXA_MO_MEDIA, FAIXA_MO_ALTA)

#: Faixa de MO adotada como padrão no cenário de estudo.
FAIXA_MO_PADRAO = FAIXA_MO_MEDIA

# ---------------------------------------------------------------------------
# Tipos de cultura antecedente reconhecidos pelo Manual
# ---------------------------------------------------------------------------
# É este eixo que carrega o mecanismo agronômico central do modelo: a dose de N
# recomendada para uma gramínea depende de QUAL cultura a antecedeu. Uma
# leguminosa antecedente reduz a dose; uma gramínea antecedente a aumenta.
ANTECESSORA_LEGUMINOSA = "leguminosa"
ANTECESSORA_GRAMINEA = "graminea"
ANTECESSORA_POUSIO = "pousio"

TIPOS_ANTECESSORA = (
    ANTECESSORA_LEGUMINOSA,
    ANTECESSORA_GRAMINEA,
    ANTECESSORA_POUSIO,
)

# Estações de cultivo (o horizonte de planejamento alterna verão/inverno)
ESTACAO_VERAO = "verao"
ESTACAO_INVERNO = "inverno"


@dataclass(frozen=True)
class Cultura:
    """Descreve uma cultura candidata do sistema de rotação.

    Os atributos agrupam-se em três blocos:

    1. Classificação agronômica (``familia``, ``fixadora_n``, ``comercial``,
       ``estacao``) — determina o papel da cultura na rotação.
    2. Exportação de nutrientes (``n_grao``, ``p2o5_grao``, ``k2o_grao``) —
       Tabela 6.1.3 do Manual, p. 106.
    3. Demanda de fertilizante (``manutencao_*`` e ``n_por_antecessora``) —
       Tabela 6.1.2 (p. 105-106) e seções por cultura (6.1.x).
    """

    nome: str
    #: 'leguminosa' | 'graminea' | 'brassicacea'
    familia: str
    #: True quando a cultura supre seu N por fixação biológica (rizóbio)
    fixadora_n: bool
    #: True para culturas de grãos com valor de mercado; False para cobertura
    comercial: bool
    #: Estação de cultivo predominante no Sul de SC
    estacao: str

    # --- Tabela 6.1.2 (p. 105-106): rendimento de referência e manutenção ---
    rendimento_ref: float
    manutencao_p2o5: float
    manutencao_k2o: float

    # --- Tabela 6.1.3 (p. 106): teor médio de nutriente no grão (kg/t) ------
    n_grao: float
    p2o5_grao: float
    k2o_grao: float

    # --- Demanda de N por faixa de MO e por cultura antecedente ------------
    # Estrutura: {faixa_mo: {tipo_antecessora: kg de N/ha}}
    # Para culturas fixadoras o Manual declara a adubação nitrogenada "não
    # recomendada", o que é representado aqui por zeros.
    n_por_antecessora: dict[str, dict[str, float]] = field(default_factory=dict)

    #: Referência de página no Manual, para rastreabilidade
    fonte: str = ""

    def demanda_n(
        self,
        faixa_mo: str = FAIXA_MO_PADRAO,
        antecessora: str = ANTECESSORA_GRAMINEA,
    ) -> float:
        """Dose de N (kg/ha) recomendada para esta cultura.

        Parameters
        ----------
        faixa_mo:
            Faixa de matéria orgânica do solo (ver :data:`FAIXAS_MO`).
        antecessora:
            Tipo da cultura que ocupou a área no período anterior.

        Notes
        -----
        Nem toda cultura tem uma coluna específica para ``pousio`` no Manual —
        apenas o milho (Tabela 6.1.14) distingue "consorciação ou pousio". Para
        as demais, adota-se a coluna de gramínea como aproximação conservadora
        (premissa do modelo, não valor do Manual).
        """
        if self.fixadora_n:
            return 0.0

        if faixa_mo not in FAIXAS_MO:
            raise ValueError(f"Faixa de MO desconhecida: {faixa_mo!r}")
        if antecessora not in TIPOS_ANTECESSORA:
            raise ValueError(f"Tipo de antecessora desconhecido: {antecessora!r}")

        tabela = self.n_por_antecessora[faixa_mo]
        if antecessora in tabela:
            return tabela[antecessora]
        # Premissa do modelo: sem coluna específica, usar o caso gramínea.
        return tabela[ANTECESSORA_GRAMINEA]

    def credito_n_leguminosa(self, faixa_mo: str = FAIXA_MO_PADRAO) -> float:
        """Economia de N (kg/ha) obtida quando a antecessora é leguminosa.

        Retorna um valor >= 0. É a diferença entre a dose exigida após uma
        gramínea e a dose exigida após uma leguminosa — exatamente o "crédito
        de N" que o modelo QUBO busca explorar ao ordenar a rotação.
        """
        return self.demanda_n(faixa_mo, ANTECESSORA_GRAMINEA) - self.demanda_n(
            faixa_mo, ANTECESSORA_LEGUMINOSA
        )

    def exportacao_n(self) -> float:
        """N (kg/ha) retirado da área na colheita, pelos grãos.

        Só culturas comerciais exportam: plantas de cobertura não são colhidas
        e sua biomassa retorna ao solo como palhada.
        """
        if not self.comercial:
            return 0.0
        return self.rendimento_ref * self.n_grao

    def saldo_n(self, faixa_mo: str = FAIXA_MO_PADRAO) -> float:
        """Saldo de N do cultivo (kg/ha) = reposição − exportação.

        Positivo significa que a área termina o cultivo com mais N do que
        começou; negativo, que houve extração líquida.

        Premissa para fixadoras
        -----------------------
        Soja, ervilhaca e tremoço têm saldo **zero** por definição. O Manual
        declara a adubação nitrogenada "não recomendada" para elas porque a
        fixação biológica supre a demanda da cultura: o N que sai no grão veio
        do ar, não do solo. Sem essa premissa a soja apareceria com saldo de
        −180 kg/ha e o modelo se recusaria a plantá-la.

        Por que a dose usada é a do pior caso
        -------------------------------------
        A reposição é tomada como a dose após gramínea, e **não** a dose real
        dada a antecessora. Não é descuido: o termo de balanço do QUBO é
        ``λ(Σ saldo)²``, e um saldo que dependesse da antecessora já é
        quadrático — elevá-lo ao quadrado produziria termos de quarta ordem,
        fora da forma QUBO. Manter o saldo linear nas variáveis é o que
        permite que o balanço seja expresso sem variáveis auxiliares.

        A consequência é que o termo de balanço enxerga **composição** (quais
        culturas, em que proporção) e não **sequência**. Quem captura a ordem
        é o crédito de N, que é naturalmente quadrático.
        """
        if self.fixadora_n:
            return 0.0
        return self.demanda_n(faixa_mo, ANTECESSORA_GRAMINEA) - self.exportacao_n()


# ---------------------------------------------------------------------------
# Catálogo de culturas
# ---------------------------------------------------------------------------
# Tabela de N para gramíneas de inverno (aveia branca / aveia preta / trigo),
# idêntica nas seções 6.1.4 (p. 117), 6.1.5 (p. 118) e 6.1.21 (p. 132).
_N_GRAMINEA_INVERNO = {
    FAIXA_MO_BAIXA: {ANTECESSORA_LEGUMINOSA: 60.0, ANTECESSORA_GRAMINEA: 80.0},
    FAIXA_MO_MEDIA: {ANTECESSORA_LEGUMINOSA: 40.0, ANTECESSORA_GRAMINEA: 60.0},
    FAIXA_MO_ALTA: {ANTECESSORA_LEGUMINOSA: 20.0, ANTECESSORA_GRAMINEA: 20.0},
}

# Milho é a única cultura com coluna explícita para "consorciação ou pousio"
# (Tabela 6.1.14, p. 125).
_N_MILHO = {
    FAIXA_MO_BAIXA: {
        ANTECESSORA_LEGUMINOSA: 70.0,
        ANTECESSORA_POUSIO: 80.0,
        ANTECESSORA_GRAMINEA: 90.0,
    },
    FAIXA_MO_MEDIA: {
        ANTECESSORA_LEGUMINOSA: 50.0,
        ANTECESSORA_POUSIO: 60.0,
        ANTECESSORA_GRAMINEA: 70.0,
    },
    FAIXA_MO_ALTA: {
        ANTECESSORA_LEGUMINOSA: 40.0,
        ANTECESSORA_POUSIO: 40.0,
        ANTECESSORA_GRAMINEA: 50.0,
    },
}

#: Sem demanda de N: fixação biológica via inoculação com rizóbio.
_N_FIXADORA = {faixa: {ANTECESSORA_GRAMINEA: 0.0} for faixa in FAIXAS_MO}


SOJA = Cultura(
    nome="Soja",
    familia="leguminosa",
    fixadora_n=True,  # 6.1.18, p. 130: adubação nitrogenada não recomendada
    comercial=True,
    estacao=ESTACAO_VERAO,
    rendimento_ref=3.0,
    manutencao_p2o5=45.0,
    manutencao_k2o=75.0,
    n_grao=60.0,
    p2o5_grao=14.0,
    k2o_grao=20.0,
    n_por_antecessora=_N_FIXADORA,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.18 (p. 130)",
)

MILHO = Cultura(
    nome="Milho",
    familia="graminea",
    fixadora_n=False,
    comercial=True,
    estacao=ESTACAO_VERAO,
    rendimento_ref=6.0,
    manutencao_p2o5=90.0,
    manutencao_k2o=60.0,
    n_grao=16.0,
    p2o5_grao=8.0,
    k2o_grao=6.0,
    n_por_antecessora=_N_MILHO,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.14 (p. 125-126)",
)

AVEIA_PRETA = Cultura(
    nome="Aveia preta",
    familia="graminea",
    fixadora_n=False,
    comercial=False,  # usada como planta de cobertura de inverno
    estacao=ESTACAO_INVERNO,
    rendimento_ref=3.0,
    manutencao_p2o5=45.0,
    manutencao_k2o=30.0,
    n_grao=20.0,
    p2o5_grao=7.0,
    k2o_grao=5.0,
    n_por_antecessora=_N_GRAMINEA_INVERNO,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.5 (p. 118)",
)

ERVILHACA = Cultura(
    nome="Ervilhaca",
    familia="leguminosa",
    fixadora_n=True,  # 6.1.10, p. 122: adubação nitrogenada não recomendada
    comercial=False,  # leguminosa de cobertura de inverno
    estacao=ESTACAO_INVERNO,
    rendimento_ref=2.0,
    manutencao_p2o5=40.0,
    manutencao_k2o=50.0,
    n_grao=35.0,
    p2o5_grao=15.0,
    k2o_grao=19.0,
    n_por_antecessora=_N_FIXADORA,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.10 (p. 122)",
)

TRIGO = Cultura(
    nome="Trigo",
    familia="graminea",
    fixadora_n=False,
    comercial=True,
    estacao=ESTACAO_INVERNO,
    rendimento_ref=3.0,
    manutencao_p2o5=45.0,
    manutencao_k2o=30.0,
    n_grao=22.0,
    p2o5_grao=10.0,
    k2o_grao=6.0,
    n_por_antecessora=_N_GRAMINEA_INVERNO,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.21 (p. 132)",
)

TREMOCO = Cultura(
    nome="Tremoço",
    familia="leguminosa",
    fixadora_n=True,  # 6.1.20, p. 132: adubação nitrogenada não recomendada
    comercial=False,
    estacao=ESTACAO_INVERNO,
    rendimento_ref=2.0,
    manutencao_p2o5=30.0,
    manutencao_k2o=50.0,
    n_grao=30.0,
    p2o5_grao=12.0,
    k2o_grao=15.0,
    n_por_antecessora=_N_FIXADORA,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.20 (p. 132)",
)

NABO_FORRAGEIRO = Cultura(
    nome="Nabo forrageiro",
    familia="brassicacea",
    fixadora_n=False,
    comercial=False,
    estacao=ESTACAO_INVERNO,
    rendimento_ref=3.0,
    manutencao_p2o5=45.0,
    manutencao_k2o=60.0,
    n_grao=20.0,
    p2o5_grao=11.0,
    k2o_grao=18.0,
    # 6.1.16 (p. 128) tabula N apenas por MO, sem distinguir antecessora.
    n_por_antecessora={
        FAIXA_MO_BAIXA: {ANTECESSORA_GRAMINEA: 60.0},
        FAIXA_MO_MEDIA: {ANTECESSORA_GRAMINEA: 50.0},
        FAIXA_MO_ALTA: {ANTECESSORA_GRAMINEA: 20.0},
    },
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 106); 6.1.16 (p. 128)",
)


# O feijão é leguminosa, mas **não** é fixadora no sentido deste modelo. O
# Manual (6.1.11, p. 123) é explícito: "em geral, a inoculação das sementes com
# rizóbio não é suficiente para suprir toda a demanda de N do feijoeiro".
# Por isso ele recebe dose de N, ao contrário de soja, ervilhaca e tremoço.
# Como antecessora continua sendo leguminosa, e portanto gera crédito de N.
_N_FEIJAO = {
    FAIXA_MO_BAIXA: {ANTECESSORA_GRAMINEA: 70.0},
    FAIXA_MO_MEDIA: {ANTECESSORA_GRAMINEA: 50.0},
    FAIXA_MO_ALTA: {ANTECESSORA_GRAMINEA: 30.0},
}

_N_SORGO = {
    FAIXA_MO_BAIXA: {ANTECESSORA_GRAMINEA: 75.0},
    FAIXA_MO_MEDIA: {ANTECESSORA_GRAMINEA: 55.0},
    FAIXA_MO_ALTA: {ANTECESSORA_GRAMINEA: 20.0},
}


FEIJAO = Cultura(
    nome="Feijão",
    familia="leguminosa",
    fixadora_n=False,  # 6.1.11, p. 123: a inoculação não supre a demanda
    comercial=True,
    estacao=ESTACAO_VERAO,
    rendimento_ref=2.0,
    manutencao_p2o5=30.0,
    manutencao_k2o=40.0,
    n_grao=50.0,
    p2o5_grao=10.0,
    k2o_grao=15.0,
    n_por_antecessora=_N_FEIJAO,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 107); 6.1.11 (p. 123)",
)

SORGO = Cultura(
    nome="Sorgo",
    familia="graminea",
    fixadora_n=False,
    comercial=True,
    estacao=ESTACAO_VERAO,
    rendimento_ref=4.0,
    manutencao_p2o5=60.0,
    manutencao_k2o=40.0,
    n_grao=15.0,
    p2o5_grao=8.0,
    k2o_grao=4.0,
    n_por_antecessora=_N_SORGO,
    fonte="Tab. 6.1.2 (p. 105-106); Tab. 6.1.3 (p. 107); 6.1.19 (p. 131)",
)


#: Todas as culturas parametrizadas, indexadas pelo nome.
CATALOGO: dict[str, Cultura] = {
    c.nome: c
    for c in (
        SOJA,
        MILHO,
        AVEIA_PRETA,
        ERVILHACA,
        TRIGO,
        TREMOCO,
        NABO_FORRAGEIRO,
        FEIJAO,
        SORGO,
    )
}


# ---------------------------------------------------------------------------
# Cenário de estudo
# ---------------------------------------------------------------------------
#: Conjunto de 4 culturas do cenário-base do TCC.
#:
#: A seleção é deliberada e cobre as duas dimensões que o modelo precisa
#: exercitar: duas estações (verão/inverno) e as duas famílias que produzem o
#: crédito de N (leguminosa antecedendo gramínea).
#:
#:   - Verão   : Soja (leguminosa, comercial) e Milho (gramínea, comercial)
#:   - Inverno : Ervilhaca (leguminosa, cobertura) e Aveia preta (gramínea,
#:               cobertura)
#:
#: A rotação clássica da região — soja/aveia preta/milho/ervilhaca — é portanto
#: representável, e o modelo pode descobrir se ela é de fato a de menor custo.
CENARIO_BASE: list[Cultura] = [SOJA, MILHO, AVEIA_PRETA, ERVILHACA]

# ---------------------------------------------------------------------------
# Escada de escalabilidade
# ---------------------------------------------------------------------------
# O estudo de escalabilidade cresce pelo **número de culturas**, mantendo o
# cenário em 1 talhão x 4 períodos. Com n culturas o modelo tem 4n variáveis
# binárias, então a escada 4..8 cobre de 16 a 32 qubits.
#
# Por que não crescer pelo número de talhões
# ------------------------------------------
# Nenhum termo do modelo liga um talhão a outro: alocação única, sequência,
# fitossanitário e balanço vivem todos dentro do mesmo talhão. Acrescentar um
# segundo talhão portanto **duplica** o problema em vez de ampliá-lo — o ótimo
# de 2 talhões é exatamente o dobro do de 1. Um cenário de 32 qubits montado
# assim seria, de fato, dois problemas independentes de 16 qubits.
#
# Crescer pelo número de culturas amplia o espaço de decisão de verdade: cada
# cultura nova acrescenta uma escolha em todos os períodos da sua estação.
#
# Por que alternar as estações a cada degrau
# ------------------------------------------
# O horizonte tem 2 períodos de verão e 2 de inverno. Enriquecer uma só
# estação deixaria a outra engessada: com apenas 2 culturas de verão, a
# penalidade de repetição sazonal já determina os dois períodos de verão, e o
# "crescimento" do problema seria ilusório. Por isso cada degrau alterna a
# estação que recebe a cultura nova.
_ESCADA: dict[int, list[Cultura]] = {
    4: [SOJA, MILHO, AVEIA_PRETA, ERVILHACA],                            # 2V + 2I
    5: [SOJA, MILHO, AVEIA_PRETA, ERVILHACA, TRIGO],                     # 2V + 3I
    6: [SOJA, MILHO, AVEIA_PRETA, ERVILHACA, TRIGO, FEIJAO],             # 3V + 3I
    7: [SOJA, MILHO, AVEIA_PRETA, ERVILHACA, TRIGO, FEIJAO, TREMOCO],    # 3V + 4I
    8: [SOJA, MILHO, AVEIA_PRETA, ERVILHACA, TRIGO, FEIJAO, TREMOCO,
        SORGO],                                                          # 4V + 4I
}

#: Conjunto completo da escada (8 culturas, 4 por estação).
CENARIO_AMPLIADO: list[Cultura] = _ESCADA[8]

#: Cenários indexados pelo número de culturas, para uso nos experimentos.
CENARIOS_POR_TAMANHO: dict[int, list[Cultura]] = dict(_ESCADA)


def validar_cenario(culturas: list[Cultura]) -> list[str]:
    """Verifica se um conjunto de culturas produz uma instância bem formulada.

    Devolve a lista de problemas encontrados; lista vazia significa que o
    cenário é admissível.
    """
    problemas: list[str] = []

    estacoes_cobertas = {c.estacao for c in culturas}
    for estacao in (ESTACAO_VERAO, ESTACAO_INVERNO):
        if estacao not in estacoes_cobertas:
            problemas.append(
                f"nenhuma cultura de {estacao}: todos os períodos dessa estação "
                "ficarão sem alocação admissível."
            )

    if not any(c.comercial for c in culturas):
        problemas.append(
            "nenhuma cultura comercial: a meta de produção não pode ser atingida."
        )

    return problemas


def resumo_culturas(culturas: list[Cultura] | None = None) -> str:
    """Devolve uma tabela textual com os parâmetros das culturas informadas."""
    culturas = culturas if culturas is not None else CENARIO_BASE

    cabecalho = (
        f"{'Cultura':<16} {'Família':<13} {'Estação':<8} {'Com.':<5} "
        f"{'N(kg/ha)':>9} {'ExpN':>6} {'SaldoN':>7} {'CréditoN':>9} "
        f"{'P2O5':>6} {'K2O':>6}"
    )
    linhas = [cabecalho, "-" * len(cabecalho)]
    for c in culturas:
        linhas.append(
            f"{c.nome:<16} {c.familia:<13} {c.estacao:<8} "
            f"{('sim' if c.comercial else 'não'):<5} "
            f"{c.demanda_n():>9.0f} {c.exportacao_n():>6.0f} "
            f"{c.saldo_n():>+7.0f} {c.credito_n_leguminosa():>9.0f} "
            f"{c.manutencao_p2o5:>6.0f} {c.manutencao_k2o:>6.0f}"
        )
    linhas.append("")
    linhas.append(
        f"Faixa de MO adotada: {FAIXA_MO_PADRAO}%  |  "
        "N(kg/ha) = dose após gramínea (pior caso)"
    )
    linhas.append("ExpN     = N exportado nos grãos (só culturas colhidas)")
    linhas.append("SaldoN   = reposicao - exportacao; fixadoras sao neutras (0)")
    linhas.append("CréditoN = economia de N quando a antecessora é leguminosa")
    linhas.append("P2O5/K2O = adubação de manutenção; fora da função objetivo,")
    linhas.append("           reportadas como métrica agronômica dos planos")
    return "\n".join(linhas)


if __name__ == "__main__":
    print("Cenário-base do TCC (4 culturas)\n")
    print(resumo_culturas(CENARIO_BASE))

    print("\n\nCenário completo da escada (8 culturas)\n")
    print(resumo_culturas(CENARIO_AMPLIADO))

    print("\n\nEscada de escalabilidade (1 talhão x 4 períodos)\n")
    print(f"{'Culturas':>9} {'Qubits':>7} {'Verão':>6} {'Inverno':>8}  Cultura acrescentada")
    print("-" * 72)
    anterior: set[str] = set()
    for n, cults in sorted(CENARIOS_POR_TAMANHO.items()):
        atual = {c.nome for c in cults}
        nova = atual - anterior
        rotulo = ", ".join(sorted(nova)) if anterior else "(cenário-base)"
        nv = sum(1 for c in cults if c.estacao == ESTACAO_VERAO)
        ni = sum(1 for c in cults if c.estacao == ESTACAO_INVERNO)
        print(f"{n:>9} {4 * n:>7} {nv:>6} {ni:>8}  {rotulo}")
        anterior = atual

    print("\n\nValidação dos cenários da escada\n")
    for n, cults in sorted(CENARIOS_POR_TAMANHO.items()):
        problemas = validar_cenario(cults)
        estado = "OK" if not problemas else "; ".join(problemas)
        print(f"  {n} culturas: {estado}")
