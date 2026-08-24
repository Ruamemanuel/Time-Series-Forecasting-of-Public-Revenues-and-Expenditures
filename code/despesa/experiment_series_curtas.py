"""
Roda LR, SVR e MLP nas 3 séries de despesa com N_Treino < 15, com hiperparâmetros
fixos (sem cross-validation) e o máximo de lags viável: n_lags = max(1,
N_Treino // 3), com normalização z-score no treino.

Atenção ao protocolo de avaliação: aqui a previsão é one-step-ahead, ou seja, a
cada passo o histórico recebe o valor REAL do conjunto de teste — diferente dos demais
experimentos, que são recursivos. Foi uma simplificação para séries muito curtas,
e essas 3 séries são justamente as excluídas da análise principal (n=16), então a
diferença não entra nos resultados reportados.

Hiperparâmetros: Ridge alpha=1.0; SVR C=1, epsilon=0.1, rbf; MLP (50,), 1000
iterações. A saída atualiza o metricas_<timestamp>.csv com LR/SVR/MLP para essas
3 séries.

Rodar com: python experiment_series_curtas.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CAMINHO_PARQUET, RESULTS_DIR, RANDOM_STATE
from utils import (
    carregar_dados, construir_series, mapear_nomes,
    filtrar_series, preparar_serie, dividir_serie,
)
from utils.metrics import calcular_metricas
from utils.windowing import criar_janela

# Séries que precisam ser completadas (N_Treino < 15)
CODIGOS_ALVO = {"501999001", "700005391", "759540000"}

# Hiperparâmetros fixos — sem CV
_MODELOS_FIXOS = {
    "LR" : Ridge(alpha=1.0),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(C=1.0, epsilon=0.1, kernel="rbf")),
    ]),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=False,
    ),
}


def _n_lags_para_serie_curta(n_treino: int) -> int:
    """Máximo de lags que deixa pelo menos ceil(2/3) das obs para treino."""
    return max(1, n_treino // 3)


def _prever_rolling(modelo, treino_norm: np.ndarray, teste_norm: np.ndarray,
                    n_lags: int) -> np.ndarray:
    """Previsão one-step-ahead: a cada passo o histórico recebe o valor real do
    conjunto de teste (não a previsão). O modelo já vem treinado em treino_norm."""
    contexto = list(treino_norm)
    preds = []
    for val_real in teste_norm:
        x = np.array(contexto[-n_lags:]).reshape(1, -1)
        pred = float(modelo.predict(x)[0])
        preds.append(pred)
        contexto.append(val_real)          # realimenta com o valor real (one-step-ahead)
    return np.array(preds)


def rodar_series_curtas() -> pd.DataFrame:
    # 1. Carrega dados
    df         = carregar_dados(CAMINHO_PARQUET)
    series_raw = construir_series(df)
    nomes      = mapear_nomes(df)
    series, _  = filtrar_series(series_raw)

    # 2. Carrega metricas existentes
    import glob
    csvs = sorted(
        c for c in glob.glob(str(RESULTS_DIR / "metricas_*.csv"))
        if "comparacao" not in Path(c).name
    )
    if not csvs:
        raise FileNotFoundError("Nenhum metricas_*.csv encontrado.")
    met_path = Path(csvs[-1])
    met_df   = pd.read_csv(met_path)
    print(f"Metricas carregadas: {met_path.name}  ({len(met_df)} linhas)")

    novos_registros = []

    for cod, serie_original in series.items():
        if cod not in CODIGOS_ALVO:
            continue

        nome = nomes.get(cod, str(cod))
        serie_pronta, grupo_sinal, _ = preparar_serie(serie_original)
        tv, tsv, tidx, tesidx, _, _ = dividir_serie(serie_pronta)

        tv  = np.asarray(tv,  dtype=float)
        tsv = np.asarray(tsv, dtype=float)
        n_treino  = len(tv)
        n_holdout = len(tsv)
        n_lags    = _n_lags_para_serie_curta(n_treino)

        # z-score sobre o treino
        mean_tr = tv.mean()
        std_tr  = tv.std() if tv.std() > 0 else 1.0
        tv_norm  = (tv  - mean_tr) / std_tr
        tsv_norm = (tsv - mean_tr) / std_tr

        X_tr, y_tr = criar_janela(tv_norm, n_lags)

        print(f"\n[{cod}] {nome[:60]}")
        print(f"  N_Treino={n_treino}  N_Holdout={n_holdout}  n_lags={n_lags}"
              f"  instâncias_treino={len(X_tr)}")

        for nome_mod, modelo_base in _MODELOS_FIXOS.items():
            import copy
            modelo = copy.deepcopy(modelo_base)

            try:
                modelo.fit(X_tr, y_tr)
                preds_norm = _prever_rolling(modelo, tv_norm, tsv_norm, n_lags)
                preds = preds_norm * std_tr + mean_tr          # desnormaliza
                metricas = calcular_metricas(tsv, preds)

                print(f"  {nome_mod}: RMSE={metricas['RMSE']:,.2f}  "
                      f"MAE={metricas['MAE']:,.2f}  MAPE={metricas['MAPE']:.2f}%")

                novos_registros.append({
                    "timestamp"   : met_df["timestamp"].iloc[0],
                    "Codigo"      : int(cod),
                    "Nome"        : nome,
                    "Grupo_Sinal" : grupo_sinal,
                    "N_Total"     : n_treino + n_holdout,
                    "N_Treino"    : n_treino,
                    "N_Holdout"   : n_holdout,
                    "N_Lags"      : n_lags,
                    "Modelo"      : nome_mod,
                    "Parametros"  : f"fixo_sem_cv  n_lags={n_lags}",
                    "RMSE"        : metricas["RMSE"],
                    "MAE"         : metricas["MAE"],
                    "MAPE"        : metricas["MAPE"],
                    "Selecionado" : False,   # reavaliado abaixo
                    "Ruido_Branco": False,
                })

            except Exception as exc:
                print(f"  {nome_mod}: ERRO — {exc}")

    if not novos_registros:
        print("Nenhum registro novo gerado.")
        return met_df

    novos_df = pd.DataFrame(novos_registros)

    # 3. Concatena com metricas existentes
    met_completo = pd.concat([met_df, novos_df], ignore_index=True)

    # 4. Reavalia Selecionado por série: menor RMSE entre todos os modelos
    met_completo["Selecionado"] = False
    for cod_val in met_completo["Codigo"].unique():
        mask = met_completo["Codigo"] == cod_val
        sub  = met_completo.loc[mask]
        # ignora ruído branco (sem previsão)
        modelaveis = sub[sub["Ruido_Branco"] == False]
        if modelaveis.empty:
            continue
        idx_melhor = modelaveis["RMSE"].idxmin()
        met_completo.loc[idx_melhor, "Selecionado"] = True

    # 5. Salva
    met_completo.to_csv(met_path, index=False, encoding="utf-8-sig")
    print(f"\nMétricas atualizadas: {met_path.name}  ({len(met_completo)} linhas)")

    # 6. Resumo comparativo
    print(f"\n{'='*70}")
    print("COMPARATIVO — SÉRIES CURTAS")
    print(f"{'='*70}")
    for cod in CODIGOS_ALVO:
        sub = met_completo[
            (met_completo["Codigo"].astype(str) == str(cod)) &
            (met_completo["Ruido_Branco"] == False)
        ][["Modelo","RMSE","N_Lags","Selecionado"]].sort_values("RMSE")
        nome = nomes.get(cod, str(cod))
        print(f"\n  {nome[:65]}")
        print(sub.to_string(index=False))
    print(f"{'='*70}\n")

    return met_completo


if __name__ == "__main__":
    rodar_series_curtas()
