"""
Configuração inicial do projeto de despesas liquidadas. Rode uma vez depois de
clonar:

    python setup.py

Confere a versão do Python (mínimo 3.9), cria as pastas data/ e results/,
verifica as dependências e o arquivo de dados, e imprime um resumo do ambiente.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

OK   = "\033[92m[OK]\033[0m"
ERR  = "\033[91m[ERR]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def checar_python():
    versao = sys.version_info
    if versao < (3, 9):
        print(f"{ERR} Python {versao.major}.{versao.minor} detectado. Mínimo requerido: 3.9")
        sys.exit(1)
    print(f"{OK} Python {versao.major}.{versao.minor}.{versao.micro}")


def criar_pastas():
    pastas = {
        ROOT / "data"   : "Pasta de dados brutos",
        ROOT / "results": "Pasta de resultados gerados",
    }
    for pasta, descricao in pastas.items():
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"{OK} {descricao}: {pasta.relative_to(ROOT)}/")


def checar_dependencias():
    dependencias = [
        "pandas", "numpy", "matplotlib",
        "statsmodels", "sklearn", "pyarrow",
        "IPython", "openpyxl",
    ]
    faltando = []
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"{OK} {dep}")
        except ImportError:
            print(f"{ERR} {dep} — não instalado")
            faltando.append(dep)

    if faltando:
        print(f"\n{WARN} Instale as dependências faltando com:")
        print(f"     pip install -r requirements.txt\n")
    else:
        print(f"\n{OK} Todas as dependências estão instaladas.")


def checar_dados():
    from config import CAMINHO_PARQUET
    if CAMINHO_PARQUET.exists():
        tamanho_mb = CAMINHO_PARQUET.stat().st_size / 1_048_576
        print(f"{OK} Arquivo de dados encontrado: {CAMINHO_PARQUET.name} ({tamanho_mb:.1f} MB)")
    else:
        print(f"{WARN} Arquivo de dados NÃO encontrado.")
        print(f"     Esperado em: {CAMINHO_PARQUET}")
        print(f"     Copie ts_liquidado.parquet para a pasta data/ antes de rodar.")


if __name__ == "__main__":
    print("\n" + "-" * 60)
    print("  SETUP - Previsao de Series Temporais de Despesas Liquidadas")
    print("-" * 60)

    print("\n[1] Verificando Python ...")
    checar_python()

    print("\n[2] Criando pastas do projeto ...")
    criar_pastas()

    print("\n[3] Verificando dependências ...")
    checar_dependencias()

    print("\n[4] Verificando arquivo de dados ...")
    checar_dados()

    print("\n" + "-" * 60)
    print("  Para rodar o experimento:")
    print("    python experiment.py")
    print("  Ou abra o notebook:")
    print("    notebooks/01_experimento_geral.ipynb")
    print("-" * 60 + "\n")
