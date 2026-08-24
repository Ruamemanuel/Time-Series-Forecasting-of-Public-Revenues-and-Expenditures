"""
LightGBM treinado por cluster: a mesma máquina do LGBM global, mas restrita aos
membros de um cluster. Fica entre o pooled (Ridge/SVR/MLP no pool) e o global
(LightGBM em todas as séries).

Frente ao pooled, ganha por usar series_id como categórica nativa (aprende o
desvio de cada série sem one-hot), trazer as features de calendário já prontas e
capturar interações lag x calendário x série via boosting. O cuidado contra
vazamento é o mesmo do global, e a previsão também é recursiva — não usa os
valores reais do conjunto de teste.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lgbm_global import (
    construir_dataset_global,
    treinar_lgbm_global,
    prever_rolling_lgbm,
    N_LAGS_GLOBAL,
)
from utils.metrics import calcular_metricas

logger = logging.getLogger(__name__)

# Tamanho mínimo do pool (series modeláveis) para treinar o LGBM cluster.
# Abaixo desse limiar o early stopping pode falhar ou o modelo overfita trivialmente.
_MIN_MEMBROS_LGBM = 3


def _nan_result(motivo: str = "") -> dict:
    return {
        "RMSE"  : float("nan"),
        "MAE"   : float("nan"),
        "MAPE"  : float("nan"),
        "params": motivo,
        "preds" : np.array([]),
    }


def treinar_avaliar_lgbm_cluster(
    membros_cl_mod  : list[str],
    treinos         : dict,
    treino_indices  : dict,
    codigo_alvo     : str,
    teste_vals      : np.ndarray,
    teste_idx       : object,            # pd.PeriodIndex do conjunto de teste
    random_state    : int = 42,
) -> dict:
    """Treina um LightGBM só com as séries modeláveis do cluster e avalia no
    conjunto de teste da série-alvo (previsão recursiva). Reaproveita o construir_dataset_
    global, o treinar_lgbm_global e o prever_rolling_lgbm. Devolve RMSE, MAE,
    MAPE, params e as previsões."""
    # cluster pequeno demais para treinar com segurança
    membros_validos = [
        c for c in membros_cl_mod
        if c in treinos and c in treino_indices
    ]
    if len(membros_validos) < _MIN_MEMBROS_LGBM:
        logger.debug(
            "[%s] LGBM cluster ignorado: apenas %d membro(s) valido(s) (min=%d)",
            codigo_alvo, len(membros_validos), _MIN_MEMBROS_LGBM,
        )
        return _nan_result(f"cluster_pequeno({len(membros_validos)})")

    if codigo_alvo not in treinos or codigo_alvo not in treino_indices:
        return _nan_result("alvo_sem_dados")

    # Constrói dataset global restrito ao cluster
    series_data = {
        cod: (treinos[cod], treino_indices[cod])
        for cod in membros_validos
    }

    try:
        X_df, y_global, norm_params, label_enc = construir_dataset_global(
            series_data, n_lags=N_LAGS_GLOBAL
        )
    except Exception as exc:
        logger.warning("[%s] construir_dataset_global falhou: %s", codigo_alvo, exc)
        return _nan_result(f"dataset_erro: {exc}")

    if len(X_df) == 0:
        return _nan_result("dataset_vazio")

    # Treina LightGBM com early stopping
    try:
        modelo, n_est = treinar_lgbm_global(
            X_df, y_global, random_state=random_state
        )
    except Exception as exc:
        logger.warning("[%s] treinar_lgbm_global falhou: %s", codigo_alvo, exc)
        return _nan_result(f"treino_erro: {exc}")

    # Previsão rolling para a série-alvo
    cod_str = str(codigo_alvo)
    if cod_str not in norm_params:
        return _nan_result("norm_params_missing")

    mean, std = norm_params[cod_str]
    try:
        series_id = int(label_enc.transform([cod_str])[0])
    except Exception:
        return _nan_result("label_enc_missing")

    try:
        preds = prever_rolling_lgbm(
            modelo,
            treinos[codigo_alvo],
            teste_vals,
            treino_indices[codigo_alvo],
            teste_idx,
            N_LAGS_GLOBAL,
            mean,
            std,
            series_id,
        )
    except Exception as exc:
        logger.warning("[%s] prever_rolling_lgbm falhou: %s", codigo_alvo, exc)
        return _nan_result(f"previsao_erro: {exc}")

    m = calcular_metricas(np.asarray(teste_vals, dtype=float), preds)
    m["params"] = f"n_est={n_est}  membros={len(membros_validos)}"
    m["preds"]  = preds
    return m
