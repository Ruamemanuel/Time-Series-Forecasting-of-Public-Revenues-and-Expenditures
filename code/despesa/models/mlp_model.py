"""
MLP (rede neural rasa) sobre janelas de lags. Mesma receita do SVR, trocando o
estimador por MLPRegressor. O early_stopping só liga quando a dobra de treino
tem pelo menos MIN_AMOSTRAS_EARLY amostras; abaixo disso o sklearn não consegue
reservar validação interna e estoura.
"""

import logging
import warnings
from itertools import product as iterproduct

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit

from utils.metrics import calcular_metricas, rmse
from utils.windowing import criar_janela

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRID_MLP, N_SPLITS_CV, RANDOM_STATE, MIN_AMOSTRAS_EARLY

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _grid_search_mlp(
    X_tr_sc: np.ndarray,
    y_tr: np.ndarray,
    y_tr_sc: np.ndarray,
    scaler_y: StandardScaler,
    n_splits: int,
) -> dict:
    """Varre o grid e devolve os hiperparâmetros de menor RMSE na validação."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    melhor_rmse   = np.inf
    melhor_params = {
        "hidden_layer_sizes": (100,),
        "alpha"             : 0.0001,
        "learning_rate_init": 0.001,
    }

    for hls, alpha, lr in iterproduct(
        GRID_MLP["hidden_layer_sizes"],
        GRID_MLP["alpha"],
        GRID_MLP["learning_rate_init"],
    ):
        rmses_cv = []
        for tr_idx, val_idx in tscv.split(X_tr_sc):
            if len(tr_idx) < 2:
                continue
            usar_early = len(tr_idx) >= MIN_AMOSTRAS_EARLY
            m = MLPRegressor(
                hidden_layer_sizes=hls,
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=500,
                random_state=RANDOM_STATE,
                early_stopping=usar_early,
                validation_fraction=0.1 if usar_early else 0.0,
            )
            m.fit(X_tr_sc[tr_idx], y_tr_sc[tr_idx])
            pred_sc = m.predict(X_tr_sc[val_idx])
            pred    = scaler_y.inverse_transform(
                pred_sc.reshape(-1, 1)
            ).ravel()
            rmses_cv.append(rmse(y_tr[val_idx], pred))

        if rmses_cv and np.mean(rmses_cv) < melhor_rmse:
            melhor_rmse   = np.mean(rmses_cv)
            melhor_params = {
                "hidden_layer_sizes": hls,
                "alpha"             : alpha,
                "learning_rate_init": lr,
            }

    logger.debug("MLP: melhor params %s  CV-RMSE=%.2f", melhor_params, melhor_rmse)
    return melhor_params


def treinar_avaliar_mlp(
    treino_vals: np.ndarray,
    teste_vals: np.ndarray,
    n_lags: int,
) -> dict:
    """Treina no treino e mede o erro no conjunto de teste com previsão recursiva.
    Devolve os params, as previsões de teste e as métricas."""
    X_tr, y_tr = criar_janela(treino_vals, n_lags)

    # scalers só com o treino, pra não vazar informação do conjunto de teste
    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_sc  = scaler_X.transform(X_tr)
    y_tr_sc  = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()

    n_splits = min(N_SPLITS_CV, len(X_tr_sc) - 1)
    params   = _grid_search_mlp(X_tr_sc, y_tr, y_tr_sc, scaler_y, n_splits)

    usar_early = len(X_tr_sc) >= MIN_AMOSTRAS_EARLY
    modelo = MLPRegressor(
        **params,
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=usar_early,
        validation_fraction=0.1 if usar_early else 0.0,
    ).fit(X_tr_sc, y_tr_sc)

    # previsão recursiva: cada passo realimenta o que foi previsto
    seq   = list(treino_vals)
    preds = []
    for step in range(len(teste_vals)):
        janela    = np.array(seq[-n_lags:]).reshape(1, -1)
        janela_sc = scaler_X.transform(janela)
        pred_sc   = modelo.predict(janela_sc)
        pred      = scaler_y.inverse_transform(
            pred_sc.reshape(-1, 1)
        ).ravel()[0]
        preds.append(pred)
        seq.append(pred)

    metricas = calcular_metricas(teste_vals, np.array(preds))
    logger.debug(
        "MLP: RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%",
        metricas["RMSE"], metricas["MAE"], metricas["MAPE"],
    )

    return {
        "params"     : str(params),
        "scaler_X"   : scaler_X,
        "scaler_y"   : scaler_y,
        "modelo"     : modelo,
        "n_lags"     : n_lags,
        "preds_teste": preds,
        **metricas,
    }


def prever_mlp(
    todos_vals: np.ndarray,
    n_lags: int,
    params: dict,
    horizonte: int,
) -> list[float]:
    """Reajusta com a série inteira e projeta `horizonte` passos à frente."""
    X_all, y_all = criar_janela(todos_vals, n_lags)
    scaler_X = StandardScaler().fit(X_all)
    scaler_y = StandardScaler().fit(y_all.reshape(-1, 1))
    usar_early = len(X_all) >= MIN_AMOSTRAS_EARLY
    modelo   = MLPRegressor(
        **params,
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=usar_early,
        validation_fraction=0.1 if usar_early else 0.0,
    ).fit(
        scaler_X.transform(X_all),
        scaler_y.transform(y_all.reshape(-1, 1)).ravel(),
    )
    seq = list(todos_vals)
    fc  = []
    for _ in range(horizonte):
        jan = scaler_X.transform(np.array(seq[-n_lags:]).reshape(1, -1))
        p   = scaler_y.inverse_transform(
            modelo.predict(jan).reshape(-1, 1)
        ).ravel()[0]
        fc.append(p)
        seq.append(p)
    return fc
