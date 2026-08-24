"""
Orquestra o pipeline em três etapas e, ao final, a comparação unificada:

  1. experiment.py        modelos individuais (ARIMA, SVR, MLP, LR)
  2. experiment_local.py  pooled por cluster + LGBM-cluster
  3. experiment_global.py Ridge / RF / MLP / LGBM globais

Uso:
  python run_all.py            roda tudo
  python run_all.py --from 2   retoma a partir da etapa 2
  python run_all.py --only 3   roda só a etapa 3
  python run_all.py --compare  só a comparação final

As etapas 2 e 3 dependem do CSV de métricas da etapa 1, então rode a etapa 1 ao
menos uma vez antes de pular para as outras. Numa máquina típica, o pipeline
inteiro leva de ~1h30 a ~3h — a etapa 1 é a mais pesada (grid search por série).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parent

# Etapas disponíveis
ETAPAS = [
    (1, "experiment.py",        "Modelos individuais  (ARIMA / SVR / MLP / LR)"),
    (2, "experiment_local.py",  "Pooled ML por cluster + LGBM cluster"),
    (3, "experiment_global.py", "Ridge / RF / MLP / LGBM globais"),
]

# Helpers

def _fmt_tempo(segundos: float) -> str:
    td = timedelta(seconds=int(segundos))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _cabecalho(texto: str) -> None:
    linha = "=" * 65
    print(f"\n{linha}")
    print(f"  {texto}")
    print(f"{linha}")


def rodar_etapa(script: str, label: str) -> bool:
    """
    Executa um script Python como subprocesso.
    Retorna True se terminou com sucesso (exit code 0).
    """
    _cabecalho(label)
    caminho = ROOT / script
    if not caminho.exists():
        print(f"  [ERRO] Script nao encontrado: {caminho}")
        return False

    t0  = time.perf_counter()
    cmd = [sys.executable, str(caminho)]
    print(f"  Comando : {' '.join(cmd)}")
    print(f"  Inicio  : {time.strftime('%H:%M:%S')}\n")

    resultado = subprocess.run(cmd, cwd=str(ROOT))

    elapsed = time.perf_counter() - t0
    ok      = resultado.returncode == 0
    status  = "OK" if ok else f"ERRO (code {resultado.returncode})"
    print(f"\n  Fim     : {time.strftime('%H:%M:%S')}  |  "
          f"Duracao : {_fmt_tempo(elapsed)}  |  Status: {status}")
    return ok


def rodar_comparacao() -> None:
    """Executa comparar_todos.py para gerar a tabela unificada."""
    _cabecalho("Comparacao unificada  (comparar_todos.py)")
    caminho = ROOT / "comparar_todos.py"
    if not caminho.exists():
        print("  [AVISO] comparar_todos.py nao encontrado. Pulando comparacao.")
        return
    subprocess.run([sys.executable, str(caminho)], cwd=str(ROOT))


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo de previsão de receitas."
    )
    parser.add_argument(
        "--from", dest="de", type=int, default=1, metavar="N",
        help="Começa a partir da etapa N (1, 2 ou 3). Default: 1"
    )
    parser.add_argument(
        "--only", dest="apenas", type=int, default=None, metavar="N",
        help="Roda apenas a etapa N."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Pula todos os experimentos e roda apenas a comparacao final."
    )
    parser.add_argument(
        "--no-compare", dest="sem_compare", action="store_true",
        help="Roda os experimentos mas NAO gera a comparacao ao final."
    )
    args = parser.parse_args()

    t_inicio = time.perf_counter()

    print("\n" + "=" * 65)
    print("  PIPELINE DE PREVISAO DE RECEITAS — run_all.py")
    print(f"  Python  : {sys.executable}")
    print(f"  Diretorio: {ROOT}")
    print("=" * 65)

    if args.compare:
        rodar_comparacao()
        return

    erros = []
    for num, script, label in ETAPAS:
        # Filtra por --from ou --only
        if args.apenas is not None and num != args.apenas:
            continue
        if args.apenas is None and num < args.de:
            print(f"\n  [Pulando etapa {num}: {label}]")
            continue

        ok = rodar_etapa(script, f"Etapa {num} — {label}")
        if not ok:
            erros.append(num)
            print(f"\n  [ATENCAO] Etapa {num} falhou. "
                  "Verifique os logs acima antes de continuar.")
            resp = input("  Continuar para a proxima etapa? [s/N]: ").strip().lower()
            if resp not in ("s", "sim", "y", "yes"):
                print("  Execucao interrompida pelo usuario.")
                break

    if not args.sem_compare:
        rodar_comparacao()

    # Resumo final
    total = time.perf_counter() - t_inicio
    _cabecalho("Resumo da execucao")
    for num, script, label in ETAPAS:
        if args.apenas is not None and num != args.apenas:
            continue
        if args.apenas is None and num < args.de:
            continue
        status = "FALHOU" if num in erros else "OK"
        print(f"  Etapa {num}: {label:<45}  [{status}]")
    print(f"\n  Tempo total: {_fmt_tempo(total)}")
    if erros:
        print(f"  Etapas com erro: {erros}")
    else:
        print("  Todas as etapas concluidas com sucesso.")


if __name__ == "__main__":
    main()
