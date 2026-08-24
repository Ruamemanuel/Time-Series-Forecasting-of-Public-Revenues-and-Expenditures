"""
Features de similaridade para clusterizar as séries. Tudo é calculado só na
janela de treino; o conjunto de teste não passa por aqui. São oito:

  log_mean_abs      escala de valor, log(1 + |média|)
  cv                coeficiente de variação (std / |média|)
  trend_slope_norm  inclinação linear normalizada pela escala
  acf_1/2/3         três primeiras autocorrelações (memória temporal)
  seasonal_strength força sazonal via STL (0 quando n < 24)
  skewness          assimetria da distribuição
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)

_FEATURE_COLS = [
    "log_mean_abs",
    "cv",
    "trend_slope_norm",
    "acf_1",
    "acf_2",
    "acf_3",
    "seasonal_strength",
    "skewness",
]


def extrair_features(treino_vals: np.ndarray) -> dict:
    """Calcula as oito features de uma série (a partir do treino) e devolve um
    dict com as colunas de _FEATURE_COLS."""
    v = np.asarray(treino_vals, dtype=float)
    n = len(v)

    # 1. Escala de valor
    mean_abs    = float(np.mean(np.abs(v)))
    log_mean_abs = float(np.log1p(mean_abs))

    # 2. Volatilidade relativa (coef. de variacao)
    std_v = float(np.std(v, ddof=1)) if n > 1 else 0.0
    denom = abs(float(np.mean(v))) + 1e-9
    cv    = std_v / denom

    # 3. Tendencia linear normalizada
    if n >= 2:
        t     = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, v, 1)[0])
        trend_slope_norm = slope / (mean_abs + 1e-9)
    else:
        trend_slope_norm = 0.0

    # 4. Primeiros 3 valores do ACF (lags 1-3)
    a1 = a2 = a3 = 0.0
    if n >= 5:
        try:
            acf_vals = acf(v, nlags=3, fft=True, missing="drop")
            a1 = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
            a2 = float(acf_vals[2]) if len(acf_vals) > 2 else 0.0
            a3 = float(acf_vals[3]) if len(acf_vals) > 3 else 0.0
        except Exception:
            pass

    # 5. Forca sazonal via STL (so se n >= 24 meses)
    seasonal_strength = 0.0
    if n >= 24:
        try:
            from statsmodels.tsa.seasonal import STL
            stl    = STL(v, period=12, robust=True).fit()
            v_seas = float(np.var(stl.seasonal))
            v_res  = float(np.var(stl.resid))
            seasonal_strength = v_seas / (v_seas + v_res + 1e-12)
        except Exception:
            seasonal_strength = 0.0

    # 6. Assimetria
    skewness = float(scipy_skew(v)) if n >= 3 else 0.0

    return {
        "log_mean_abs"     : log_mean_abs,
        "cv"               : cv,
        "trend_slope_norm" : trend_slope_norm,
        "acf_1"            : a1,
        "acf_2"            : a2,
        "acf_3"            : a3,
        "seasonal_strength": seasonal_strength,
        "skewness"         : skewness,
    }


def extrair_features_todas(treinos: dict) -> pd.DataFrame:
    """Extrai as features de todas as séries de {codigo: treino_vals} e devolve um
    DataFrame indexado por código (NaN vira 0.0)."""
    records = {}
    for cod, vals in treinos.items():
        try:
            records[cod] = extrair_features(np.asarray(vals, dtype=float))
        except Exception as exc:
            logger.warning("feature_extraction falhou em %s: %s", cod, exc)
            records[cod] = {k: 0.0 for k in _FEATURE_COLS}

    df = pd.DataFrame.from_dict(records, orient="index", columns=_FEATURE_COLS)
    df = df.fillna(0.0)
    logger.info("Features extraidas: %d series x %d features", *df.shape)
    return df
