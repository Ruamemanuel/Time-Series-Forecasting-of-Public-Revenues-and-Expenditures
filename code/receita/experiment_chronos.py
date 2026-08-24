"""
Acrescenta o Chronos (amazon/chronos-t5-small) ao paradigma global. Carrega as
séries com os mesmos utilitários do experiment_global, roda o Chronos de forma
recursiva em cada série, pega o comparacao_global mais recente e adiciona as
colunas Global_Chronos_*, salvando um CSV/XLSX com novo timestamp. Não re-treina
nada do que já existe — só soma o Chronos.

Rodar com: python experiment_chronos.py
"""

from __future__ import annotations

import glob
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CAMINHO_PARQUET, RESULTS_DIR, N_LAGS_GLOBAL
from utils import (
    carregar_dados, construir_series, mapear_nomes,
    filtrar_series, preparar_serie, dividir_serie,
)
from utils.acf_analysis import analisar_serie
from utils.metrics import calcular_metricas
from models.chronos_global import carregar_pipeline, prever_rolling_chronos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment_chronos")


def _carregar_comparacao_global() -> tuple[pd.DataFrame, Path]:
    csvs = sorted(glob.glob(str(RESULTS_DIR / "comparacao_global_*.csv")))
    if not csvs:
        raise FileNotFoundError(
            "Nenhum comparacao_global_*.csv encontrado. "
            "Execute experiment_global.py primeiro."
        )
    path = Path(csvs[-1])
    logger.info("Carregando resultados globais existentes: %s", path.name)
    return pd.read_csv(path), path


def rodar_chronos():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Carrega e prepara series
    logger.info("Carregando dados: %s", CAMINHO_PARQUET)
    df         = carregar_dados(CAMINHO_PARQUET)
    series_raw = construir_series(df)
    nomes      = mapear_nomes(df)
    series, _  = filtrar_series(series_raw)
    logger.info("Series validas: %d", len(series))

    treinos : dict = {}
    testes  : dict = {}

    for cod, serie_original in series.items():
        serie_pronta, _, _ = preparar_serie(serie_original)
        tv, tsv, _, _, _, _ = dividir_serie(serie_pronta)
        treinos[cod] = np.asarray(tv, dtype=np.float32)
        testes[cod]  = np.asarray(tsv, dtype=np.float32)

    # Usa exatamente as series que os outros modelos globais ja previram
    comp_existente, _ = _carregar_comparacao_global()
    codigos_global = set(comp_existente["Codigo"].astype(str))
    series_alvo = [cod for cod in series if str(cod) in codigos_global]
    logger.info("Series alvo (mesmas do comparacao_global): %d", len(series_alvo))

    # 2. Carrega Chronos (uma unica vez)
    logger.info("Carregando Chronos (amazon/chronos-t5-small)...")
    pipeline = carregar_pipeline()
    logger.info("Chronos carregado.")

    # 3. previsão recursiva por série
    resultados_chronos: dict[str, dict] = {}
    n_total = len(series_alvo)

    for i, cod in enumerate(series_alvo, 1):
        nome = nomes.get(cod, str(cod))
        tv   = treinos[cod]
        tsv  = testes[cod]

        try:
            preds    = prever_rolling_chronos(pipeline, tv, tsv,
                                              context_length=N_LAGS_GLOBAL)
            metricas = calcular_metricas(tsv.astype(float), preds)
            resultados_chronos[str(cod)] = {
                "Global_Chronos_RMSE": metricas["RMSE"],
                "Global_Chronos_MAE" : metricas["MAE"],
                "Global_Chronos_MAPE": metricas["MAPE"],
            }
            logger.info("[%d/%d] %s — RMSE=%.2f", i, n_total, nome, metricas["RMSE"])
        except Exception as exc:
            logger.warning("[%d/%d] %s falhou: %s", i, n_total, nome, exc)
            resultados_chronos[str(cod)] = {
                "Global_Chronos_RMSE": float("nan"),
                "Global_Chronos_MAE" : float("nan"),
                "Global_Chronos_MAPE": float("nan"),
            }

    # 4. Mescla com comparacao_global existente
    comp_df = comp_existente.copy()

    rows_chronos = []
    for cod_str, metricas in resultados_chronos.items():
        rows_chronos.append({"Codigo": int(cod_str), **metricas})
    df_chronos = pd.DataFrame(rows_chronos)

    # Remove colunas Chronos pre-existentes (se houver re-execucao)
    for col in ["Global_Chronos_RMSE", "Global_Chronos_MAE", "Global_Chronos_MAPE"]:
        if col in comp_df.columns:
            comp_df = comp_df.drop(columns=[col])

    comp_df = comp_df.merge(df_chronos, on="Codigo", how="left")

    # 5. Exporta
    csv_path  = RESULTS_DIR / f"comparacao_global_{timestamp}.csv"
    xlsx_path = RESULTS_DIR / f"comparacao_global_{timestamp}.xlsx"

    comp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        comp_df.to_excel(writer, sheet_name="Global", index=False)

    logger.info("Salvo: %s", csv_path.name)
    print(f"\nCSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

    # 6. Resumo
    rmse_vals = comp_df["Global_Chronos_RMSE"].dropna()
    ind_vals  = comp_df.loc[comp_df["Global_Chronos_RMSE"].notna(), "Ind_RMSE"]
    wins      = int((comp_df["Global_Chronos_RMSE"] < comp_df["Ind_RMSE"]).sum())
    n_valid   = int(rmse_vals.notna().sum())

    print(f"\n{'='*60}")
    print("CHRONOS — RESUMO")
    print(f"{'='*60}")
    print(f"  Series avaliadas : {n_valid}")
    print(f"  RMSE mediano     : {rmse_vals.median():,.2f}")
    print(f"  Wins vs Individual: {wins}/{n_valid}")
    print(f"{'='*60}\n")

    return comp_df


if __name__ == "__main__":
    rodar_chronos()
