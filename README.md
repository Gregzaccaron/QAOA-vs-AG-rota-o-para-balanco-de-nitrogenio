# QAOA e algoritmos genéticos na rotação de culturas

Implementação e dados do trabalho **"Comparação entre o algoritmo quântico QAOA
e algoritmos genéticos na otimização do balanço de nitrogênio em rotação de
culturas na região da AMESC"** (Gregori Rodrigues Zaccaron, Ciência da
Computação, UNESC, 2026).

O problema é escolher que cultura plantar em cada período de um horizonte de
rotação, minimizando a necessidade de fertilizante nitrogenado e respeitando
restrições agronômicas. Ele é formulado como **QUBO** e resolvido por dois
métodos sobre **exatamente a mesma matriz** `Q`: o QAOA, em simulador quântico,
e um algoritmo genético. Esta pasta contém tudo o que é necessário para
reproduzir os 480 experimentos, as tabelas e as figuras do artigo.

---

## 1. Começando

```bash
python -m pip install -r requirements.txt
```

Para conferir que a instalação está completa, imprima a escada de instâncias:

```bash
python parametros.py
```

A saída lista as cinco instâncias, de 4 a 8 culturas (16 a 32 variáveis
binárias), e qual cultura entra em cada degrau.

Para um teste rápido, que termina em poucos segundos e não usa o QAOA:

```bash
python experimento_comparativo.py --execucoes 3 --sem-qaoa --ags padrao viavel
```

---

## 2. Reproduzindo os resultados publicados

Este é o comando exato que gerou `resultados/comparativo.csv`:

```bash
python experimento_comparativo.py \
    --execucoes 30 \
    --ags padrao viavel \
    --mixers X XY \
    --qaoa-p 2 \
    --qaoa-maxiter 100 \
    --qaoa-reinicios 3 \
    --qaoa-shots 4096 \
    --max-qubits-qaoa 24 \
    --threads 8 \
    --seed-base 42
```

**Atenção aos padrões.** `--qaoa-maxiter` vale 150 por omissão e o artigo usou
100; `--ags` e `--mixers` trazem só uma variante cada por omissão, e o
delineamento fatorial exige as duas de cada. Sem esses argumentos explícitos o
resultado não é o publicado.

| | |
|---|---|
| Tempo total | cerca de **9 horas** na máquina descrita em `requirements.txt` |
| Onde o tempo vai | 97% nas instâncias de 24 variáveis do QAOA (386 s e 557 s por execução) |
| Memória de pico | 268 MB para 24 qubits; 6,16 GB para 28 |
| Retomada | o CSV é escrito de forma incremental; se a execução for interrompida, rodar o mesmo comando continua de onde parou |

A varredura do número de camadas é separada, e foi feita em quatro execuções:

```bash
for p in 1 2 3 4; do
  python experimento_comparativo.py --execucoes 5 --qaoa-p $p \
      --mixers X XY --max-qubits-qaoa 20 \
      --saida resultados/varredura_p$p.csv \
      --log resultados/varredura_p.log
done
```

Não há como desligar o algoritmo genético: ele roda junto, mas custa
milissegundos e as linhas dele nesses arquivos simplesmente não são usadas por
`graficos.py`, que filtra pelas colunas `metodo` e `qaoa_p`.

Depois dos experimentos, para as figuras e a planilha:

```bash
python graficos.py        # gera resultados/figuras/*.pdf e *.png
python exportar_xlsx.py   # gera resultados/comparativo.xlsx
```

---

## 3. O que cada arquivo faz

Os módulos ficam todos no mesmo diretório, com importações planas. **Não mova
os arquivos para subpastas** sem ajustar os `import`.

| Arquivo | Papel |
|---|---|
| `parametros.py` | Catálogo de culturas e parâmetros agronômicos, extraídos do Manual de Calagem e Adubação para RS e SC (2016), com tabela e página de origem anotadas em cada valor. Define a escada de instâncias. Executável, imprime o catálogo. |
| `qubo_rotacao.py` | Constrói a matriz `Q`. Contém a classe `Pesos`, com os sete termos da função objetivo, e as rotinas de validação da formulação. É o único lugar onde o modelo é definido. |
| `algoritmo_genetico.py` | Laço evolutivo. Duas variantes, escolhidas por `preserva_viabilidade` na `ConfiguracaoAG`. |
| `qaoa_rotacao.py` | Circuito QAOA, conversão QUBO para Hamiltoniano de Ising, os dois operadores de mistura e o laço híbrido com COBYLA. |
| `qaoa_diagonal.py` | Distribuição de probabilidade exata do estado final, aproveitando que o Hamiltoniano de custo é diagonal na base computacional. É de onde saem a probabilidade do ótimo e o fator de amplificação. |
| `experimento_comparativo.py` | Orquestra o experimento: percorre instâncias, métodos e repetições, e grava o CSV. É o ponto de entrada. |
| `escalabilidade.py` | Mede tempo e memória por instância, sem otimizar nada. Gera `resultados/escalabilidade.{csv,json}`. |
| `progresso.py` | Barra de progresso e contagem exata de avaliações da função objetivo. |
| `graficos.py` | As figuras do artigo. Legíveis em preto e branco. |
| `exportar_xlsx.py` | Converte o CSV em `.xlsx` de duas abas. Sem dependência externa. |

### Por que o algoritmo genético é próprio, e não PyGAD ou DEAP

Porque a **contagem de avaliações da função objetivo precisa ser exata**: é ela
que permite comparar dois métodos cujo custo por iteração difere em seis ordens
de grandeza. Numa implementação própria essa contagem fica explícita no código
e pode ser auditada (ver `progresso.ContadorDeAvaliacoes`).

---

## 4. O modelo

**Cenário:** uma área, quatro períodos de cultivo alternando verão e inverno ao
longo de dois anos agrícolas. A escala cresce pelo número de culturas
candidatas, de quatro a oito, logo `4 × n_culturas` variáveis binárias.

A variável `x[i,j,t]` vale 1 se a cultura `i` ocupa a área `j` no período `t`.
A função objetivo é `min x' Q x + c`, com `Q` triangular superior incluindo a
diagonal, o que é válido porque `x² = x` para `x` binário.

**Os sete termos**, com os pesos padrão de `qubo_rotacao.Pesos`:

| Termo | Significado | Peso |
|---|---|---|
| Custo de nitrogênio | Dose de N por alocação, descontado o crédito da cultura antecedente. **Único termo que depende da ordem das culturas.** | `custo_n = 3.00` |
| Balanço do ciclo | Desvio quadrático do saldo de N acumulado, em relação ao equilíbrio | `penal_balanco_n = 0.03` |
| Alocação única | Exatamente uma cultura por área e período | `penal_alocacao_unica = 600` |
| Repetição de cultura | Mesma cultura em períodos consecutivos | `penal_repeticao_cultura = 150` |
| Repetição sazonal | Mesma cultura na mesma estação de anos consecutivos (`t` e `t+2`) | `penal_repeticao_estacional = 300` |
| Mesma família botânica | Famílias iguais em períodos consecutivos | `penal_mesma_familia = 50` |
| Adequação de estação | Cultura de verão em período de inverno, e vice-versa | `penal_estacao = 300` |

Fósforo e potássio são **calculados e reportados, mas não otimizados**:
`custo_p2o5` e `custo_k2o` valem zero. As doses de manutenção desses nutrientes
são fixas por cultura no Manual, de modo que não carregam informação de
sequência; com peso diferente de zero, respondiam por 54% do valor da função
objetivo sem informar nada sobre a rotação. Os campos permanecem no código para
que reativá-los seja um ajuste de peso, não uma mudança de estrutura.

### As quatro configurações comparadas

O delineamento é fatorial 2x2: paradigma computacional contra conhecimento da
restrição de alocação única embutido nos operadores de busca.

| | Desinformado | Informado |
|---|---|---|
| **Clássico** | `--ags padrao`: cruzamento uniforme, mutação por inversão de bit, população inicial em `{0,1}^n` | `--ags viavel`: cruzamento por blocos, mutação por troca, população inicial entre planos válidos |
| **Quântico** | `--mixers X`: `B = Σ Xa`, estado inicial em superposição uniforme | `--mixers XY`: `B = Σ (XaXb + YaYb)/2` por período, estado inicial `W` por bloco |

As duas variantes informadas restringem a busca ao subespaço viável **por
construção do operador**, e não por filtragem posterior. Essa correspondência é
o que torna o contraste interpretável.

### Validação da formulação

Antes de qualquer comparação, `qubo_rotacao.py` verifica duas coisas a cada
alteração de termo ou de peso:

1. **Por enumeração exaustiva**, que o mínimo global sobre todo o espaço
   `{0,1}^n` coincide com o mínimo sobre as soluções agronomicamente viáveis.
2. **Analiticamente**, que cada penalidade supera a maior economia que a
   violação da restrição correspondente poderia proporcionar
   (`ModeloQUBO.verificar_penalidades`).

Isso não é zelo excessivo. Numa formulação irrestrita, uma penalidade
subdimensionada desloca o mínimo global para uma solução inviável e barata, que
uma metaheurística bem-sucedida encontra. O erro é silencioso: nada falha, e os
números parecem bons. Numa calibração anterior, com a penalidade de repetição
sazonal em 200, o ótimo do cenário de cinco culturas era **monocultura sazonal
de soja**, porque o plano pagava a penalidade e ainda saía mais barato.

---

## 5. Os dados

`resultados/comparativo.csv` tem 480 linhas, uma por execução, e 27 colunas.
Formato canônico: vírgula separa colunas, ponto separa decimais. Para abrir no
Excel use o `.xlsx` gerado por `exportar_xlsx.py`.

| Coluna | Conteúdo |
|---|---|
| `instancia`, `n_culturas`, `n_talhoes`, `n_periodos`, `n_variaveis` | Identificação da instância |
| `metodo`, `execucao`, `instante` | Configuração, número da repetição e horário |
| `custo`, `custo_otimo`, `desvio_percentual`, `atingiu_otimo` | Qualidade. `custo` é o resultado obtido **com o orçamento de 4096 medições**, não a probabilidade exata |
| `alocacao_valida` | Se a solução satisfaz todas as restrições agronômicas |
| `demanda_n_kg_ha`, `saldo_n_kg_ha`, `adubacao_p2o5_kg_ha`, `adubacao_k2o_kg_ha` | Grandezas agronômicas do plano, em kg/ha no horizonte |
| `tempo_s`, `n_avaliacoes`, `geracoes`, `geracao_convergencia` | Custo computacional e esforço de busca |
| `qaoa_p`, `valor_esperado`, `profundidade_circuito`, `n_portas`, `prob_melhor`, `memoria_estimada_gb` | Específicas do QAOA. `prob_melhor` é a probabilidade **exata**, sem ruído de amostragem |

Os demais arquivos: `varredura_p1..4.csv` (efeito do número de camadas),
`historicos.json` (trajetórias de convergência, usadas por `graficos.py`),
`escalabilidade.{csv,json}` (tempo e memória por instância) e
`figuras/` (as figuras em PDF e PNG).

---

## 6. O que esperar

Reproduzindo o comando da seção 2, estes são os números publicados. Desvio
percentual em relação ao ótimo global, obtido por enumeração exaustiva.

| Qubits | Configuração | Desvio médio | Ótimos | Válidas | Tempo |
|---|---|---|---|---|---|
| 16 | AG padrão | 2,340% | 21/30 | 30/30 | 0,0072 s |
| 16 | AG viável | **0,000%** | 30/30 | 30/30 | 0,0290 s |
| 16 | QAOA mixer X | 21,499% | 11/30 | 30/30 | 2,48 s |
| 16 | QAOA mixer XY | 0,520% | 28/30 | 30/30 | 4,79 s |
| 20 | AG padrão | 2,889% | 19/30 | 30/30 | 0,0085 s |
| 20 | AG viável | **0,000%** | 30/30 | 30/30 | 0,0299 s |
| 20 | QAOA mixer X | 131,029% | 0/30 | 24/30 | 13,38 s |
| 20 | QAOA mixer XY | 0,069% | 28/30 | 30/30 | 20,17 s |
| 24 | AG padrão | 1,548% | 27/30 | 30/30 | 0,0103 s |
| 24 | AG viável | **0,000%** | 30/30 | 30/30 | 0,0301 s |
| 24 | QAOA mixer X | 361,644% | 0/30 | 13/30 | 386,09 s |
| 24 | QAOA mixer XY | 0,516% | 29/30 | 30/30 | 557,23 s |
| 28 | AG padrão | 7,263% | 24/30 | 30/30 | 0,0122 s |
| 28 | AG viável | **0,000%** | 30/30 | 30/30 | 0,0328 s |
| 32 | AG padrão | 3,863% | 25/30 | 30/30 | 0,0117 s |
| 32 | AG viável | **0,000%** | 30/30 | 30/30 | 0,0318 s |

O desvio de 361,6% do mixer X **não mede má qualidade agronômica, mede
penalidade**: o método não consegue permanecer no conjunto de alocações válidas.
Sob a restrição de alocação única, todo vizinho de um plano válido é inválido
para um operador que inverte um bit, e o subespaço viável é só 0,0077% do
espaço total em 24 variáveis.

Comparadas as duas configurações informadas, **AG viável e QAOA mixer XY, a
diferença de qualidade não é estatisticamente significativa** (Mann-Whitney
bilateral, p entre 0,16 e 0,33). A diferença está no tempo: 18.513 vezes em 24
variáveis, em simulação ideal e sem ruído, que é a condição mais favorável
possível ao método quântico.

---

## 7. Avisos de reprodutibilidade

**O algoritmo genético reproduz bit a bit.** Mesma semente, mesmo resultado.

**O QAOA não reproduz entre versões diferentes de código.** Constatou-se que
uma perturbação de ordem `1e-15` no valor esperado da energia, originada de
alteração na decomposição do circuito que **não modifica o operador
implementado**, basta para levar o COBYLA a um mínimo local distinto, com
diferença de dezenas de pontos percentuais no desvio final. Com o código fixo o
efeito é reprodutível, inclusive variando o número de *threads*; mas qualquer
mudança que altere o arredondamento o desloca, e o efeito se intensifica com o
tamanho da instância. **Resultados de QAOA obtidos com versões diferentes de
código não são comparáveis entre si**, ainda que o modelo matemático seja o
mesmo. É a razão pela qual as versões exatas das bibliotecas estão anotadas em
`requirements.txt`.

**A instância de 32 qubits não roda o QAOA.** O vetor de estado exigiria 64 GiB
(`2³² × 16` bytes) contra os 32 GB disponíveis. A de 28 roda, com 6,16 GB de
pico, mas as 30 repetições levam cerca de 25 horas, e por isso o artigo reporta
o QAOA só até 24 variáveis.

**A formulação é separável por área.** Nenhum termo da função objetivo liga uma
área a outra, de modo que `k` áreas constituem `k` subproblemas independentes.
Um cenário de 32 qubits construído por replicação de áreas seria, na prática,
dois problemas de 16 qubits. É por isso que a escada cresce pelo número de
culturas, e não pelo de áreas.

---

## 8. Fonte dos parâmetros agronômicos

Todos os valores de nitrogênio vêm do **Manual de Calagem e Adubação para os
Estados do Rio Grande do Sul e de Santa Catarina** (CQFS-RS/SC, 2016), com
tabela e página anotadas em `parametros.py`. Dados complementares sobre plantas
de cobertura vêm do **Guia Prático de Plantas de Cobertura** (ESALQ/USP, 2024).

O mecanismo agronômico central é o **crédito de nitrogênio por cultura
antecedente**: o Manual recomenda dose menor de N quando a cultura anterior é
leguminosa do que quando é gramínea. É o único dado do Manual que enxerga a
*ordem* das culturas, e é o que torna o problema combinatório em vez de uma
simples escolha independente por período.

Uma cultura rompe a dicotomia de propósito: o **feijão é leguminosa mas recebe
adubação nitrogenada**, porque, como registra o Manual, a inoculação das
sementes com rizóbio em geral não supre toda a sua demanda. Como antecedente,
porém, segue gerando crédito para a sucessora.
