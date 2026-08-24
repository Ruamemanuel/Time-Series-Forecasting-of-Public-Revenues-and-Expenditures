"""
Estudo de ablação do experimento de despesas liquidadas, isolando o efeito de
dois componentes: o número de lags (3, 6, 12) e o tamanho do conjunto de teste (3, 6, 12
meses). Os modelos são avaliados com a mesma previsão recursiva do experimento
principal, mantendo o resto fixo.

Rodar com: python ablation.py
"""

import logging
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CAMINHO_PARQUET, RESULTS_DIR,
    ABLATION_LAGS, ABLATION_HOLDOUTS,
    LR_ALPHA, MIN_TREINO,
)
from utils import (
    carregar_dados, construir_series, mapear_nomes,
    filtrar_series, preparar_serie,
)
from utils.acf_analysis import analisar_serie
from models.arima_model import treinar_avaliar_arima
from models.lr_model    import treinar_avaliar_lr
from models.svr_model   import treinar_avaliar_svr
from models.mlp_model   import treinar_avaliar_mlp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _carregar_series(caminho=CAMINHO_PARQUET):
    df         = carregar_dados(caminho)
    series_raw = construir_series(df)
    series, _  = filtrar_series(series_raw)
    nomes      = mapear_nomes(df)
    return series, nomes


def _avaliar_todos(treino, teste, n_lags, n_treino) -> list[dict]:
    """Avalia todos os modelos e retorna lista de registros."""
    resultados = []
    rodar_ml   = n_treino >= MIN_TREINO

    try:
        r = treinar_avaliar_arima(treino, teste)
        resultados.append({"Modelo": "ARIMA", **{k: r[k] for k in ["RMSE","MAE","MAPE"]}})
    except Exception:
        pass

    if rodar_ml:
        for fn, nome_mod in [
            (lambda t, te: treinar_avaliar_lr(t, te, n_lags, LR_ALPHA), "LR"),
            (lambda t, te: treinar_avaliar_svr(t, te, n_lags),          "SVR"),
            (lambda t, te: treinar_avaliar_mlp(t, te, n_lags),          "MLP"),
        ]:
            try:
                r = fn(treino, teste)
                resultados.append({"Modelo": nome_mod, **{k: r[k] for k in ["RMSE","MAE","MAPE"]}})
            except Exception:
                pass

    return resultados


def _salvar_csv(df: pd.DataFrame, nome: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = RESULTS_DIR / f"{nome}_{ts}.csv"
    df.to_csv(caminho, index=False)
    logger.info("Salvo: %s (%d linhas)", caminho.name, len(df))
    return caminho


def ablation_lags(caminho=CAMINHO_PARQUET, verbose=True) -> pd.DataFrame:
    """
    Para cada série e cada valor de N_LAGS em ABLATION_LAGS,
    avalia LR, SVR e MLP mantendo o conjunto de teste adaptativo fixo.
    ARIMA não usa lags mas é incluído como referência.
    """
    logger.info("Ablação de lags: iniciando ...")
    series, nomes = _carregar_series(caminho)
    registros     = []

    for idx, (codigo, serie_original) in enumerate(series.items(), 1):
        serie_pronta, grupo, _ = preparar_serie(serie_original)
        vals    = serie_pronta.values.astype(float)
        n_total = len(vals)

        n_holdout = max(3, min(12, int(n_total * 0.2)))
        n_treino  = n_total - n_holdout

        if n_treino < 5:
            continue

        treino = vals[:-n_holdout]
        teste  = vals[-n_holdout:]

        acf_info = analisar_serie(treino)
        if acf_info["e_ruido_branco"]:
            if verbose:
                print(f"  [{idx}/{len(series)}] Ablação de lags: {codigo} - RUÍDO BRANCO, pulando")
            continue

        for n_lags in ABLATION_LAGS:
            if n_lags >= n_treino:
                continue

            resultados = _avaliar_todos(treino, teste, n_lags, n_treino)
            for r in resultados:
                registros.append({
                    "Codigo"       : codigo,
                    "Nome"         : nomes.get(codigo, "—"),
                    "Grupo_Sinal"  : grupo,
                    "N_Total"      : n_total,
                    "N_Treino"     : n_treino,
                    "N_Holdout"    : n_holdout,
                    "N_Lags"       : n_lags,
                    "LB_Pvalor"    : acf_info["lb_pvalor"],
                    "Lags_Sig_PACF": str(acf_info["lags_sig_pacf"]),
                    **r,
                })

    df = pd.DataFrame(registros)
    _salvar_csv(df, "ablation_lags")
    logger.info("Ablação de lags: concluído (%d registros)", len(df))
    return df


def ablation_holdout(caminho=CAMINHO_PARQUET, verbose=True) -> pd.DataFrame:
    """
    Para cada série e cada valor de N_HOLDOUT em ABLATION_HOLDOUTS,
    avalia todos os modelos mantendo lags adaptativos fixos.
    """
    logger.info("Ablação do conjunto de teste: iniciando ...")
    series, nomes = _carregar_series(caminho)
    registros     = []

    for idx, (codigo, serie_original) in enumerate(series.items(), 1):
        serie_pronta, grupo, _ = preparar_serie(serie_original)
        vals    = serie_pronta.values.astype(float)
        n_total = len(vals)

        for n_holdout in ABLATION_HOLDOUTS:
            n_treino = n_total - n_holdout

            if n_treino < 5 or n_holdout >= n_total:
                continue

            treino = vals[:-n_holdout]
            teste  = vals[-n_holdout:]

            acf_info = analisar_serie(treino)
            if acf_info["e_ruido_branco"]:
                continue

            n_lags = acf_info["n_lags"]

            resultados = _avaliar_todos(treino, teste, n_lags, n_treino)
            for r in resultados:
                registros.append({
                    "Codigo"       : codigo,
                    "Nome"         : nomes.get(codigo, "—"),
                    "Grupo_Sinal"  : grupo,
                    "N_Total"      : n_total,
                    "N_Treino"     : n_treino,
                    "N_Holdout"    : n_holdout,
                    "N_Lags"       : n_lags,
                    "LB_Pvalor"    : acf_info["lb_pvalor"],
                    "Lags_Sig_PACF": str(acf_info["lags_sig_pacf"]),
                    "Metodo_Lags"  : acf_info["metodo_lags"],
                    **r,
                })

        if verbose:
            print(f"  [{idx}/{len(series)}] Ablação do conjunto de teste: {codigo}")

    df = pd.DataFrame(registros)
    _salvar_csv(df, "ablation_holdout")
    logger.info("Ablação do conjunto de teste: concluído (%d registros)", len(df))
    return df


def rodar_ablation_completo(caminho=CAMINHO_PARQUET, verbose=True) -> dict:
    """Executa os dois ablation studies em sequência."""
    logger.info("=== ABLATION STUDY ===")
    lags_df    = ablation_lags(caminho, verbose)
    holdout_df = ablation_holdout(caminho, verbose)
    logger.info("=== ABLATION CONCLUÍDO ===")
    return {"lags_df": lags_df, "holdout_df": holdout_df}


if __name__ == "__main__":
    resultados = rodar_ablation_completo(verbose=True)

    for nome, df in resultados.items():
        print(f"\n-- {nome} ({len(df)} linhas) ──")
        if not df.empty:
            print(df.groupby(["Modelo"])[["RMSE","MAE","MAPE"]].mean().round(2).to_string())
