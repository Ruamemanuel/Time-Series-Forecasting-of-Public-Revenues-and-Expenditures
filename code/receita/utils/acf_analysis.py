"""
Análise de autocorrelação do treino (nunca do conjunto de teste, para não vazar) com dois
objetivos:

  1. Detectar ruído branco com o Ljung-Box. Se o teste não rejeita H0
     (p >= ACF_ALPHA), as autocorrelações são todas nulas, nenhum modelo extrai
     padrão e a série é pulada.
  2. Escolher o número de lags pelo PACF, que mede a correlação direta entre y_t
     e y_{t-k} descontando os lags intermediários — ou seja, "até onde ainda há
     informação nova". O ACF inflaria a escolha: um AR(1) forte tem ACF
     significativa em vários lags, mas só o lag 1 carrega informação direta.

Se o Ljung-Box rejeita ruído branco mas o PACF não acusa nenhum lag, recorre a um
fallback heurístico (uma fração do tamanho do treino). Os parâmetros (MAX_LAGS,
ACF_ALPHA, etc.) ficam no config.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy import stats as sp_stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MAX_LAGS, MIN_LAGS, PCT_LAGS,
    ACF_ALPHA, LB_NLAGS_RATIO, LB_NLAGS_MIN, LB_NLAGS_MAX,
)

logger = logging.getLogger(__name__)

# Tenta statsmodels; usa fallback scipy se não instalado
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox as _sm_ljungbox
    from statsmodels.tsa.stattools import pacf as _sm_pacf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.debug("statsmodels não disponível — usando implementações scipy/numpy")


# PACF (scipy fallback via Yule-Walker)

def _pacf_yw(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    PACF via equações de Yule-Walker (sem statsmodels).
    Retorna array[0..nlags]; pacf[0] = 1.0 por definição.
    """
    n  = len(x)
    xm = x - x.mean()
    # Autocovariâncias
    gamma = np.array([np.dot(xm[:n-k], xm[k:]) / n for k in range(nlags + 1)])
    if gamma[0] == 0:
        return np.zeros(nlags + 1)

    pacf_vals = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        # Yule-Walker: resolve sistema k×k
        R = np.array([[gamma[abs(i - j)] for j in range(k)] for i in range(k)])
        r = gamma[1:k + 1]
        try:
            phi = np.linalg.solve(R, r)
            pacf_vals[k] = phi[-1]
        except np.linalg.LinAlgError:
            pacf_vals[k] = 0.0
    return pacf_vals


def _calcular_pacf(treino: np.ndarray) -> tuple[np.ndarray, float]:
    """
    PACF do treino até MAX_LAGS e banda de confiança 95% (± 1.96/√n).
    Usa statsmodels (método 'ywm') se disponível; fallback para Yule-Walker numpy.
    """
    n     = len(treino)
    nlags = min(MAX_LAGS, n // 2 - 1)   # PACF: limite n//2 para estabilidade
    nlags = max(nlags, 1)

    if _HAS_STATSMODELS:
        pacf_vals = _sm_pacf(treino, nlags=nlags, method="ywm")
    else:
        pacf_vals = _pacf_yw(treino, nlags)

    banda = 1.96 / np.sqrt(n)
    return pacf_vals, banda


def _lags_significativos_pacf(pacf_vals: np.ndarray, banda: float) -> List[int]:
    """Índices 1..n onde |PACF| > banda (exclui lag 0)."""
    return [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > banda]


# Ljung-Box (baseado em ACF)

def _acf_manual(x: np.ndarray, nlags: int) -> np.ndarray:
    """ACF normalizada até nlags (fallback sem statsmodels)."""
    n   = len(x)
    xm  = x - x.mean()
    var = np.dot(xm, xm)
    if var == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(xm[:n-k], xm[k:]) / var for k in range(nlags + 1)])


def _ljung_box_scipy(x: np.ndarray, h: int) -> float:
    """
    Ljung-Box via chi² (fallback sem statsmodels).

    Retorna o p-valor do teste conjunto com h graus de liberdade —
    equivalente ao teste Ljung-Box padrão em h lags.
    NÃO usa min() para evitar inflação do erro Tipo I por comparações múltiplas.
    """
    n   = len(x)
    xm  = x - x.mean()
    var = np.dot(xm, xm) / n
    if var == 0:
        return 1.0
    acf_sq = _acf_manual(x, h)
    # Estatística Q acumulada até lag h (chi² com h graus de liberdade)
    Q_h = n * (n + 2) * sum(acf_sq[j] ** 2 / (n - j) for j in range(1, h + 1))
    return float(sp_stats.chi2.sf(Q_h, df=h))


def _ljung_box_pvalor(treino: np.ndarray) -> float:
    """
    Menor p-valor do Ljung-Box com h lags adaptativos.
    h = clip(round(n × LB_NLAGS_RATIO), LB_NLAGS_MIN, LB_NLAGS_MAX)
    Retorna np.nan em caso de falha.
    """
    n = len(treino)
    h = int(np.clip(round(n * LB_NLAGS_RATIO), LB_NLAGS_MIN, LB_NLAGS_MAX))
    h = min(h, n - 2)
    if h < 1:
        return np.nan
    try:
        if _HAS_STATSMODELS:
            # lags=[h] → um único teste conjunto com h graus de liberdade.
            # NÃO usar lags=h (int) com .min(): isso testaria lags 1..h
            # individualmente e inflacionaria o erro Tipo I (comparações múltiplas).
            resultado = _sm_ljungbox(treino, lags=[h], return_df=True)
            return float(resultado["lb_pvalue"].iloc[-1])
        else:
            return _ljung_box_scipy(treino, h)
    except Exception as exc:
        logger.debug("Ljung-Box falhou: %s", exc)
        return np.nan


# Fallback heurístico

def _n_lags_fallback(n_treino: int) -> int:
    """min(MAX_LAGS, max(MIN_LAGS, int(n_treino × PCT_LAGS)))"""
    return max(MIN_LAGS, min(MAX_LAGS, int(n_treino * PCT_LAGS)))


# Interface pública

def analisar_serie(treino_vals: np.ndarray) -> dict:
    """Roda Ljung-Box (ruído branco) e PACF (seleção de lags) sobre o treino.

    Devolve um dict com: e_ruido_branco (bool), motivo_exclusao, n_lags (0 se for
    ruído branco), lags_sig_pacf, pacf_vals, banda (±1.96/√n), lb_pvalor e
    metodo_lags ('pacf', 'fallback_heuristico' ou 'nenhum')."""
    n = len(treino_vals)

    # 1. Ljung-Box — diagnóstico de ruído branco
    lb_pvalor = _ljung_box_pvalor(treino_vals)

    # 2. PACF — lags com memória direta significativa
    pacf_vals, banda = _calcular_pacf(treino_vals)
    lags_sig_pacf    = _lags_significativos_pacf(pacf_vals, banda)

    # 3. Decisão: ruído branco?
    # Critério primário: LB não rejeita H0 (p >= alpha)
    # Fallback (LB indisponível): ausência total de lags PACF significativos
    if not np.isnan(lb_pvalor):
        eh_ruido = lb_pvalor >= ACF_ALPHA
        motivo   = (
            f"Ljung-Box p={lb_pvalor:.4f} >= {ACF_ALPHA} "
            f"(não rejeita H0: ruído branco)"
        ) if eh_ruido else ""
    else:
        eh_ruido = len(lags_sig_pacf) == 0
        motivo   = (
            "Nenhum lag PACF significativo e Ljung-Box indisponível"
        ) if eh_ruido else ""

    if eh_ruido:
        logger.debug("Ruído branco detectado: %s", motivo)
        return dict(
            e_ruido_branco  = True,
            motivo_exclusao = motivo,
            n_lags          = 0,
            lags_sig_pacf   = lags_sig_pacf,
            pacf_vals       = pacf_vals,
            banda           = banda,
            lb_pvalor       = lb_pvalor,
            metodo_lags     = "nenhum",
        )

    # 4. Seleção de lags pelo PACF
    if lags_sig_pacf:
        # Janela = maior lag PACF significativo (inclui toda a memória direta)
        n_lags = min(max(lags_sig_pacf), MAX_LAGS)
        metodo = "pacf"
        logger.debug(
            "Lags PACF sig.: %s → n_lags=%d (banda=±%.4f)",
            lags_sig_pacf, n_lags, banda,
        )
    else:
        # LB rejeita RB mas nenhum lag PACF individualmente significativo
        # (estrutura sutil) → fallback heurístico conservador
        n_lags = _n_lags_fallback(n)
        metodo = "fallback_heuristico"
        logger.debug(
            "LB rejeita RB mas PACF sem lags sig → fallback n_lags=%d", n_lags
        )

    return dict(
        e_ruido_branco  = False,
        motivo_exclusao = "",
        n_lags          = n_lags,
        lags_sig_pacf   = lags_sig_pacf,
        pacf_vals       = pacf_vals,
        banda           = banda,
        lb_pvalor       = lb_pvalor,
        metodo_lags     = metodo,
    )
