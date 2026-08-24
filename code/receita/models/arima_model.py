"""
ARIMA com seleção automática de ordem. O grau de diferenciação d vem do teste
ADF (até d=2) e os graus p e q saem de uma busca pelo menor AIC no treino.
A avaliação no conjunto de teste usa origem fixa: ajusta uma vez no treino e prevê os h
passos de uma vez, sem olhar nenhum valor real do teste — é o que se faz na
prática ao projetar 12 meses à frente.
"""

import logging
import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from utils.metrics import calcular_metricas

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _testar_estacionariedade(serie: np.ndarray, alpha: float = 0.05) -> bool:
    """Retorna True se a série for estacionária pelo teste ADF."""
    resultado = adfuller(pd.Series(serie).dropna())
    return resultado[1] < alpha


def _determinar_d(treino: np.ndarray, max_d: int = 2) -> int:
    """Determina o grau de diferenciação via ADF (máximo max_d)."""
    d = 0
    serie = treino.copy()
    while not _testar_estacionariedade(serie) and d < max_d:
        d += 1
        serie = np.diff(serie)
    return d


def _grid_search_arima(treino: np.ndarray, d: int) -> tuple[int, int, float]:
    """Busca p e q em [0,3] pelo menor AIC e devolve (p, q, aic)."""
    melhor_aic = np.inf
    p_best, q_best = 1, 1

    for p, q in product(range(4), range(4)):
        try:
            mod = ARIMA(treino, order=(p, d, q)).fit()
            if mod.aic < melhor_aic:
                melhor_aic = mod.aic
                p_best, q_best = p, q
        except Exception:
            continue

    return p_best, q_best, melhor_aic


def treinar_avaliar_arima(
    treino_vals: np.ndarray,
    teste_vals: np.ndarray,
) -> dict:
    """Seleciona a ordem, ajusta no treino e prevê o conjunto de teste em origem fixa.
    Devolve a ordem (p,d,q), as previsões de teste e as métricas. O conjunto de teste
    nunca é usado no ajuste."""
    d = _determinar_d(treino_vals)
    p, q, aic = _grid_search_arima(treino_vals, d)
    logger.debug("ARIMA: ordem selecionada (%d,%d,%d)  AIC=%.2f", p, d, q, aic)

    # origem fixa: um ajuste no treino e h passos de uma vez, sem ver o teste
    h = len(teste_vals)
    try:
        mod   = ARIMA(treino_vals, order=(p, d, q)).fit()
        preds = mod.forecast(steps=h).tolist()
    except Exception as exc:
        logger.warning("ARIMA forecast falhou, usando ultimo valor: %s", exc)
        preds = [float(treino_vals[-1])] * h

    metricas = calcular_metricas(teste_vals, np.array(preds))
    logger.debug(
        "ARIMA: RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%",
        metricas["RMSE"], metricas["MAE"], metricas["MAPE"],
    )

    return {
        "params"    : f"({p},{d},{q})",
        "d"         : d,
        "p"         : p,
        "q"         : q,
        "preds_teste": preds,
        **metricas,
    }


def prever_arima(
    todos_vals: np.ndarray,
    p: int, d: int, q: int,
    horizonte: int,
) -> list[float]:
    """Reajusta com a série inteira e projeta `horizonte` passos à frente."""
    mod = ARIMA(todos_vals, order=(p, d, q)).fit()
    return mod.forecast(steps=horizonte).tolist()
