"""Acompanhamento visual de execucoes longas, no console e em arquivo de log.

Por que existe
--------------
O experimento roda por horas, desacompanhado. Sem retorno visual nao se
distingue "esta trabalhando" de "travou", e nao se sabe se vale esperar. Este
modulo resolve isso com duas saidas que se complementam:

- **no console**, uma linha que se reescreve no lugar, com barra, contagem,
  tempo decorrido e previsao de termino;
- **em arquivo**, uma linha por evento concluido, gravada e descarregada na
  hora, para poder abrir depois e para sobreviver a um desligamento.

Quando a saida nao e um terminal (redirecionada para arquivo, por exemplo), a
reescrita no lugar e desligada automaticamente, senao o arquivo encheria de
caracteres de controle.
"""

from __future__ import annotations

import pathlib
import sys
import time
from datetime import datetime, timedelta


def formatar_duracao(segundos: float) -> str:
    """Duracao legivel e curta: 45s, 12m08s, 3h41m."""
    if segundos < 0 or segundos != segundos:          # negativo ou NaN
        return "--"
    segundos = int(segundos)
    if segundos < 60:
        return f"{segundos}s"
    if segundos < 3600:
        return f"{segundos // 60}m{segundos % 60:02d}s"
    return f"{segundos // 3600}h{(segundos % 3600) // 60:02d}m"


class Progresso:
    """Barra de progresso com log persistente.

    Parameters
    ----------
    total:
        Numero de etapas previstas. Se nao for conhecido, passe ``None``: a
        barra vira um contador sem percentual.
    rotulo:
        Nome curto do que esta rodando, mostrado no inicio da linha.
    arquivo_log:
        Caminho do log. Cada etapa concluida vira uma linha, com data e hora.
    largura:
        Largura da barra, em caracteres.
    """

    def __init__(
        self,
        total: int | None,
        rotulo: str = "",
        arquivo_log: str | pathlib.Path | None = None,
        largura: int = 24,
        pesos: list[float] | None = None,
    ) -> None:
        self.total = total
        self.rotulo = rotulo
        self.largura = largura
        # Custo esperado de cada etapa, na ordem em que serao executadas. Uma
        # lista de tamanho diferente de `total` seria erro de quem chamou, e
        # aqui e simplesmente ignorada: uma previsao de termino e enfeite e
        # nunca deve derrubar um experimento de horas.
        self.pesos = list(pesos) if pesos else None
        if self.pesos is not None and total is not None                 and len(self.pesos) != total:
            self.pesos = None
        self._peso_total = sum(self.pesos) if self.pesos else 0.0
        self.feito = 0
        self.t_inicio = time.perf_counter()
        self.detalhe = ""
        self._ultimo_desenho = 0.0
        self._interativo = sys.stdout.isatty()
        self._tam_linha = 0

        self._log = None
        if arquivo_log is not None:
            caminho = pathlib.Path(arquivo_log)
            caminho.parent.mkdir(parents=True, exist_ok=True)
            self._log = caminho.open("a", encoding="utf-8")
            self.registrar(f"=== inicio: {rotulo or 'execucao'} ===")

    # -- log ---------------------------------------------------------------

    def registrar(self, texto: str) -> None:
        """Escreve uma linha no log, com data e hora, e descarrega na hora.

        O descarregamento imediato e o que garante que o arquivo esteja
        legivel enquanto a execucao continua, e que nada se perca se a maquina
        for desligada no meio.
        """
        if self._log is None:
            return
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.write(f"[{agora}] {texto}\n")
        self._log.flush()

    # -- console -----------------------------------------------------------

    def _prever_restante(self, decorrido: float) -> float:
        """Segundos que faltam, ponderados pelo custo esperado das etapas.

        Sem pesos a previsao e ``decorrido / feito * faltam``, que supoe
        etapas de custo igual. Quando elas diferem por ordens de grandeza --
        no experimento, de centesimos de segundo a dezenas de minutos -- essa
        media anuncia "faltam 2s" com duas horas pela frente. Com os pesos a
        regra de tres usa o *trabalho* ja feito, nao a contagem de etapas.
        """
        if self.pesos is None or self._peso_total <= 0:
            return decorrido / self.feito * (self.total - self.feito)
        feito = sum(self.pesos[: self.feito])
        if feito <= 0:
            return 0.0
        return decorrido / feito * (self._peso_total - feito)

    def _linha(self) -> str:
        decorrido = time.perf_counter() - self.t_inicio
        partes = []
        if self.rotulo:
            partes.append(f"[{self.rotulo}]")

        if self.total:
            fracao = self.feito / self.total
            cheio = int(fracao * self.largura)
            barra = "#" * cheio + "-" * (self.largura - cheio)
            partes.append(barra)
            partes.append(f"{self.feito}/{self.total}")
            partes.append(f"{fracao * 100:3.0f}%")
        else:
            partes.append(f"{self.feito} feitos")

        partes.append(formatar_duracao(decorrido))

        if self.total and self.feito > 0:
            restante = self._prever_restante(decorrido)
            partes.append(f"faltam {formatar_duracao(restante)}")
            fim = datetime.now() + timedelta(seconds=restante)
            partes.append(f"termina {fim.strftime('%H:%M')}")

        if self.detalhe:
            partes.append(self.detalhe)
        return "  ".join(partes)

    def desenhar(self, forcar: bool = False) -> None:
        """Redesenha a linha, no maximo duas vezes por segundo.

        O limite evita que o proprio desenho consuma tempo em lacos rapidos, e
        evita piscar a tela sem necessidade.
        """
        agora = time.perf_counter()
        if not forcar and agora - self._ultimo_desenho < 0.5:
            return
        self._ultimo_desenho = agora
        linha = self._linha()
        if self._interativo:
            # apaga os restos da linha anterior, que pode ser mais longa
            preenchimento = " " * max(0, self._tam_linha - len(linha))
            sys.stdout.write("\r" + linha + preenchimento)
            self._tam_linha = len(linha)
        else:
            sys.stdout.write(linha + "\n")
        sys.stdout.flush()

    # -- avanco ------------------------------------------------------------

    def atualizar(self, detalhe: str = "") -> None:
        """Atualiza o texto auxiliar sem contar uma etapa concluida."""
        if detalhe:
            self.detalhe = detalhe
        self.desenhar()

    def passo(self, detalhe: str = "", registrar: bool = True) -> None:
        """Conta uma etapa concluida."""
        self.feito += 1
        if detalhe:
            self.detalhe = detalhe
        self.desenhar(forcar=True)
        if registrar:
            alvo = f"/{self.total}" if self.total else ""
            self.registrar(f"{self.feito}{alvo}  {detalhe or self.rotulo}")

    def encerrar(self, nota: str = "") -> None:
        decorrido = time.perf_counter() - self.t_inicio
        if self._interativo:
            sys.stdout.write("\r" + " " * self._tam_linha + "\r")
            sys.stdout.flush()
        resumo = (f"{self.rotulo or 'execucao'}: {self.feito} etapas em "
                  f"{formatar_duracao(decorrido)}")
        if nota:
            resumo += f"  {nota}"
        print(resumo)
        self.registrar(f"=== fim: {resumo} ===")
        if self._log is not None:
            self._log.close()
            self._log = None

    # -- uso como contexto -------------------------------------------------

    def __enter__(self) -> "Progresso":
        self.desenhar(forcar=True)
        return self

    def __exit__(self, *_exc) -> None:
        if self._log is not None:
            self.encerrar()


class ContadorDeAvaliacoes:
    """Mostra o avanco *dentro* de uma execucao do QAOA.

    Serve para o caso que motivou todo este modulo: uma unica execucao de 28
    qubits leva horas, e sem isto a linha de progresso ficaria parada todo
    esse tempo, indistinguivel de um travamento. Este contador e chamado a
    cada avaliacao da funcao de custo e informa a execucao-mae.
    """

    def __init__(self, pai: Progresso, total_previsto: int,
                 prefixo: str = "") -> None:
        self.pai = pai
        self.total = total_previsto
        self.prefixo = prefixo
        self.n = 0
        self.melhor = float("inf")
        self.reinicio = 0

    def nova_tentativa(self, indice: int, de: int) -> None:
        self.reinicio = indice
        self.total_reinicios = de

    def __call__(self, valor: float) -> None:
        self.n += 1
        if valor < self.melhor:
            self.melhor = valor
        texto = f"{self.prefixo}aval {self.n}/{self.total}"
        if self.melhor < float("inf"):
            texto += f" melhor {self.melhor:.2f}"
        self.pai.atualizar(texto)
