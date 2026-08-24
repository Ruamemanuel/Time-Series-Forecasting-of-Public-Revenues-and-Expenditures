"""
Para cada série de despesa, gera o gráfico de painel duplo no estilo de
results/plots/serie_*.png (reaproveita reports.plots.plotar_serie):

  Painel de cima: treino e teste reais, a previsão no conjunto de teste de todos os
  candidatos (ARIMA, LR, SVR, MLP e LightGBM global) e o forecast de 12 meses do
  melhor modelo.
  Painel de baixo: resíduos do melhor modelo.

O melhor de cada série sai entre os individuais e o LightGBM global (menor RMSE
no conjunto de teste). Os arquivos levam o sufixo "_com_global".
"""
from __future__ import annotations
import sys, ast, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (CAMINHO_PARQUET, HORIZONTE, MESES_PREV, N_LAGS_GLOBAL,
                    MIN_TREINO, COR_MODELO)
from utils import (carregar_dados, construir_series, mapear_nomes,
                   filtrar_series, preparar_serie, reverter_shift, dividir_serie)
from utils.acf_analysis import analisar_serie
from utils.metrics import calcular_metricas
from models.arima_model import treinar_avaliar_arima, prever_arima
from models.lr_model    import treinar_avaliar_lr,    prever_lr
from models.svr_model   import treinar_avaliar_svr,   prever_svr
from models.mlp_model   import treinar_avaliar_mlp,   prever_mlp
from models.lgbm_global import (construir_dataset_global, treinar_lgbm_global,
                                prever_rolling_lgbm)
from reports.plots import plotar_serie

LGBM_KEY = "LGBM_glob"
COR_MODELO[LGBM_KEY] = "#16A085"   # teal, distinto do verde do forecast

df         = carregar_dados(CAMINHO_PARQUET)
series_raw = construir_series(df)
nomes      = mapear_nomes(df)
series, _  = filtrar_series(series_raw)

# usa as séries modeláveis (as que entraram no experimento individual)
import glob
unif = sorted(glob.glob(str(ROOT / "results" / "comparacao_unificada_*.csv")))[-1]
codes_alvo = set(pd.read_csv(unif)["Codigo"].astype(str))

# prepara séries e treina LightGBM global uma vez
prep, series_data = {}, {}
for cod, s in series.items():
    if str(cod) not in codes_alvo:
        continue
    sp, grupo, shift = preparar_serie(s)
    tv, tsv, tidx, tesidx, n_hold, _ = dividir_serie(sp)
    prep[str(cod)] = dict(sp=sp, grupo=grupo, shift=shift, tv=tv, tsv=tsv,
                          tidx=tidx, tesidx=tesidx, n_hold=n_hold)
    series_data[str(cod)] = (tv, tidx)

print("Treinando LightGBM global...")
Xg, yg, norm_params, label_enc = construir_dataset_global(series_data, N_LAGS_GLOBAL)
modelo_lgbm, _ = treinar_lgbm_global(Xg, yg)
print("LightGBM global treinado.\n")

fut_pidx   = pd.PeriodIndex([pd.Period(m, freq="M") for m in MESES_PREV])
meses_prev = list(fut_pidx)

for cod, p in prep.items():
    nome = nomes.get(cod, cod)
    tv, tsv = p["tv"], p["tsv"]
    n_treino = len(tv)
    acf = analisar_serie(tv)
    n_lags = acf["n_lags"] if not acf["e_ruido_branco"] else p["n_hold"]
    todos = p["sp"].values.astype(float)

    resultados = {}
    forecasts  = {}   # modelo -> fc_vals (12) na escala serie_pronta
    try:
        # modelos individuais
        r = treinar_avaliar_arima(tv, tsv); resultados["ARIMA"] = r
        forecasts["ARIMA"] = prever_arima(todos, r["p"], r["d"], r["q"], HORIZONTE)
        if n_treino >= MIN_TREINO:
            r = treinar_avaliar_lr(tv, tsv, n_lags, alpha=None); resultados["LR"] = r
            forecasts["LR"] = prever_lr(todos, n_lags, r["alpha"], HORIZONTE)
            r = treinar_avaliar_svr(tv, tsv, n_lags); resultados["SVR"] = r
            forecasts["SVR"] = prever_svr(todos, n_lags, ast.literal_eval(r["params"]), HORIZONTE)
            r = treinar_avaliar_mlp(tv, tsv, n_lags); resultados["MLP"] = r
            forecasts["MLP"] = prever_mlp(todos, n_lags, ast.literal_eval(r["params"]), HORIZONTE)

        # LightGBM global
        mean, std = norm_params[cod]; sid = int(label_enc.transform([cod])[0])
        preds_g = np.asarray(prever_rolling_lgbm(
            modelo_lgbm, tv, tsv, p["tidx"], p["tesidx"], N_LAGS_GLOBAL, mean, std, sid), dtype=float)
        met_g = calcular_metricas(tsv, preds_g)
        resultados[LGBM_KEY] = {"preds_teste": preds_g.tolist(), **met_g}
        todos_idx = p["tidx"].append(p["tesidx"])
        forecasts[LGBM_KEY] = np.asarray(prever_rolling_lgbm(
            modelo_lgbm, todos, np.zeros(HORIZONTE), todos_idx, fut_pidx,
            N_LAGS_GLOBAL, mean, std, sid), dtype=float).tolist()

        # melhor modelo e forecast revertido
        melhor = min(resultados, key=lambda k: resultados[k]["RMSE"])
        fc_orig = reverter_shift(np.asarray(forecasts[melhor], dtype=float), p["shift"]).tolist()

        plotar_serie(
            codigo=cod, nome=nome, grupo_sinal=p["grupo"],
            treino_idx=p["tidx"], treino_vals=tv,
            teste_idx=p["tesidx"], teste_vals=tsv,
            resultados_serie=resultados, melhor_modelo=melhor,
            meses_prev=meses_prev, fc_vals=fc_orig,
            n_holdout=p["n_hold"], n_lags=n_lags,
            mostrar=False, sufixo="_com_global",
        )
        print(f"[OK] {cod}  melhor={melhor}")
    except Exception as exc:
        print(f"[FALHA] {cod}: {exc}")

print("\nGraficos salvos em results/plots/  (sufixo _com_global)")
