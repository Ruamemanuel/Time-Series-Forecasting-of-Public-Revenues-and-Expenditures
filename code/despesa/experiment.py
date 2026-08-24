"""
Orquestrador do experimento individual (um modelo por série de despesa liquidada):
ARIMA, LR (Ridge), SVR e MLP.

Para cada série: classifica o sinal e trata negativas (shift), separa treino e
conjunto de teste, treina e avalia os quatro modelos, escolhe o de menor RMSE, reajusta o
vencedor em treino+conjunto de teste e projeta os 12 meses seguintes (revertendo o shift
quando houver). Ao final salva métricas, parâmetros e previsões em CSV (com
timestamp) e os gráficos por série em results/plots/.

Rodar com: python experiment.py
"""

import ast
import logging
import pickle
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
    CAMINHO_PARQUET, RESULTS_DIR, PLOTS_DIR,
    HORIZONTE, MESES_PREV, FIM_SERIE, MIN_TREINO, LR_ALPHA,
)
from utils import (
    carregar_dados, construir_series, mapear_nomes,
    filtrar_series, preparar_serie, reverter_shift,
    dividir_serie, calcular_lags,
)
from utils.acf_analysis import analisar_serie
from models import (
    treinar_avaliar_arima, prever_arima,
    treinar_avaliar_lr,    prever_lr,
    treinar_avaliar_svr,   prever_svr,
    treinar_avaliar_mlp,   prever_mlp,
)
from reports.plots import plotar_serie
from reports.excel_export import gerar_excel_resultados

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")

# Checkpoint
_CHECKPOINT_VERSION = "v1_liquidado_individual_multistep"
_CHECKPOINT_PATH    = RESULTS_DIR / "checkpoint_individual.pkl"

def _salvar_checkpoint(registros_met, registros_params, registros_prev, timestamp):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "registros_met"    : registros_met,
        "registros_params" : registros_params,
        "registros_prev"   : registros_prev,
        "timestamp"        : timestamp,
        "concluidos"       : {str(r["Codigo"]) for r in registros_met},
        "schema_version"   : _CHECKPOINT_VERSION,
    }
    with open(_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(data, f)

def _carregar_checkpoint():
    if not _CHECKPOINT_PATH.exists():
        return None
    try:
        with open(_CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        versao = data.get("schema_version", "v0")
        if versao != _CHECKPOINT_VERSION:
            logger.warning(
                "Checkpoint com schema '%s' (esperado '%s'). Iniciando do zero.",
                versao, _CHECKPOINT_VERSION,
            )
            return None
        logger.info(
            "Checkpoint carregado: %d séries já concluídas.",
            len(data["concluidos"]),
        )
        return data
    except Exception as exc:
        logger.warning("Falha ao carregar checkpoint: %s. Iniciando do zero.", exc)
        return None


def _fmt(v: float) -> str:
    if pd.isna(v):        return "N/A"
    if abs(v) >= 1e9:     return f"R$ {v/1e9:.2f} Bi"
    if abs(v) >= 1e6:     return f"R$ {v/1e6:.2f} Mi"
    if abs(v) >= 1e3:     return f"R$ {v/1e3:.2f} K"
    return f"R$ {v:,.2f}"


def rodar_experimento(
    caminho_parquet: Path = CAMINHO_PARQUET,
    salvar_resultados: bool = True,
    gerar_graficos: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Executa o experimento completo de previsão de despesas liquidadas.

    Returns
    -------
    dict com: previsoes, metricas_df, params_df, prev_df
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tenta retomar checkpoint
    ckpt = _carregar_checkpoint()
    if ckpt:
        timestamp        = ckpt["timestamp"]
        registros_met    = ckpt["registros_met"]
        registros_params = ckpt["registros_params"]
        registros_prev   = ckpt["registros_prev"]
        concluidos       = ckpt["concluidos"]
    else:
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        registros_met    = []
        registros_params = []
        registros_prev   = []
        concluidos       = set()

    # 1. Dados
    logger.info("Carregando dados de %s", caminho_parquet)
    df          = carregar_dados(caminho_parquet)
    series_raw  = construir_series(df)
    nomes       = mapear_nomes(df)
    series, exc = filtrar_series(series_raw)

    logger.info("Séries válidas: %d | Excluídas: %d", len(series), len(exc))
    if exc and verbose:
        print(f"\n  Séries excluídas ({len(exc)}):")
        for cod, motivo in exc[:10]:
            print(f"    {cod}: {motivo}")
        if len(exc) > 10:
            print(f"    ... e mais {len(exc)-10}")

    # 2. Loop por série
    previsoes = {}

    total = len(series)
    for idx, (codigo, serie_original) in enumerate(series.items(), 1):

        # Pula séries já concluídas (checkpoint)
        if str(codigo) in concluidos:
            if verbose:
                print(f"[{idx}/{total}] Fonte {codigo} — já concluída, pulando.")
            continue

        nome = nomes.get(codigo, "—")
        if verbose:
            print(f"\n[{idx}/{total}] Fonte {codigo}")
            print(f"  Nome   : {nome[:80]}")
            print(f"  Periodo: {serie_original.index.min()} -> {serie_original.index.max()}")
            print(f"  Obs.   : {len(serie_original)}")

        # 2a. Tratamento de sinal
        serie_pronta, grupo_sinal, shift_value = preparar_serie(serie_original)
        if verbose:
            print(f"  Sinal  : {grupo_sinal}"
                  + (f"  (shift={shift_value:.2f})" if shift_value else ""))

        # 2b. Divisão treino/teste
        (treino_vals, teste_vals,
         treino_idx, teste_idx,
         n_holdout, n_lags) = dividir_serie(serie_pronta)

        n_treino = len(treino_vals)
        rodar_ml = n_treino >= MIN_TREINO

        if verbose:
            print(f"  Treino : {str(treino_idx[0])} -> {str(treino_idx[-1])} "
                  f"({n_treino} obs.)  |  Conjunto de teste: {n_holdout}")
            if not rodar_ml:
                print(f"  [WARN] LR/SVR/MLP desabilitados (treino < {MIN_TREINO} obs.)")

        # 2c. Análise de ACF: ruído branco + seleção de lags
        acf_info = analisar_serie(treino_vals)
        n_lags   = acf_info["n_lags"] if not acf_info["e_ruido_branco"] else n_holdout

        if verbose:
            status_acf = (
                f"LB p={acf_info['lb_pvalor']:.4f}  "
                f"lags_sig_pacf={acf_info['lags_sig_pacf']}  "
                f"n_lags={n_lags}  método={acf_info['metodo_lags']}"
            )
            print(f"  PACF   : {status_acf}")

        # Se ruído branco → registra e pula os modelos
        if acf_info["e_ruido_branco"]:
            if verbose:
                print(f"  [WARN] RUIDO BRANCO — série pulada: {acf_info['motivo_exclusao']}")

            registros_met.append({
                "timestamp"      : timestamp,
                "Codigo"         : codigo,
                "Nome"           : nome,
                "Grupo_Sinal"    : grupo_sinal,
                "N_Total"        : len(serie_original),
                "N_Treino"       : n_treino,
                "N_Holdout"      : n_holdout,
                "N_Lags"         : 0,
                "Modelo"         : "N/A",
                "Parametros"     : "N/A",
                "RMSE"           : np.nan,
                "MAE"            : np.nan,
                "MAPE"           : np.nan,
                "Selecionado"    : False,
                "Ruido_Branco"   : True,
                "LB_Pvalor"      : acf_info["lb_pvalor"],
                "Lags_Sig_PACF"  : str(acf_info["lags_sig_pacf"]),
                "Metodo_Lags"    : acf_info["metodo_lags"],
            })
            concluidos.add(str(codigo))
            _salvar_checkpoint(registros_met, registros_params, registros_prev, timestamp)
            continue

        resultados_serie = {}

        # ARIMA
        logger.info("[%s] Treinando ARIMA ...", codigo)
        res_arima = treinar_avaliar_arima(treino_vals, teste_vals)
        resultados_serie["ARIMA"] = res_arima
        if verbose:
            print(f"  ARIMA  {res_arima['params']:15s}  "
                  f"RMSE={res_arima['RMSE']:>14,.2f}  "
                  f"MAPE={res_arima['MAPE']:.2f}%")

        # Regressão Linear
        if rodar_ml:
            logger.info("[%s] Treinando LR ...", codigo)
            # alpha=None → busca CV sobre GRID_RIDGE_ALPHAS (mesmo protocolo
            # dos modelos pooled e global — comparação justa entre paradigmas)
            res_lr = treinar_avaliar_lr(treino_vals, teste_vals, n_lags, alpha=None)
            resultados_serie["LR"] = res_lr
            if verbose:
                print(f"  LR     {res_lr['params']:15s}  "
                      f"RMSE={res_lr['RMSE']:>14,.2f}  "
                      f"MAPE={res_lr['MAPE']:.2f}%")

        # SVR
        if rodar_ml:
            logger.info("[%s] Treinando SVR ...", codigo)
            res_svr = treinar_avaliar_svr(treino_vals, teste_vals, n_lags)
            resultados_serie["SVR"] = res_svr
            if verbose:
                print(f"  SVR    {str(res_svr['params'])[:15]:15s}  "
                      f"RMSE={res_svr['RMSE']:>14,.2f}  "
                      f"MAPE={res_svr['MAPE']:.2f}%")

        # MLP
        if rodar_ml:
            logger.info("[%s] Treinando MLP ...", codigo)
            res_mlp = treinar_avaliar_mlp(treino_vals, teste_vals, n_lags)
            resultados_serie["MLP"] = res_mlp
            if verbose:
                print(f"  MLP    {str(res_mlp['params'])[:15]:15s}  "
                      f"RMSE={res_mlp['RMSE']:>14,.2f}  "
                      f"MAPE={res_mlp['MAPE']:.2f}%")

        # Seleção do melhor modelo
        melhor = min(resultados_serie, key=lambda k: resultados_serie[k]["RMSE"])
        if verbose:
            print(f"  [OK] Melhor: {melhor}  "
                  f"(RMSE={resultados_serie[melhor]['RMSE']:,.2f})")

        # Previsão final (multi-step-ahead)
        todos_vals = serie_pronta.values.astype(float)
        meses_prev = [pd.Period(m, freq="M") for m in MESES_PREV]

        if melhor == "ARIMA":
            r       = resultados_serie["ARIMA"]
            fc_vals = prever_arima(todos_vals, r["p"], r["d"], r["q"], HORIZONTE)

        elif melhor == "LR":
            # Usa o alpha ótimo encontrado pela busca CV no treino
            alpha_lr_otimo = resultados_serie["LR"]["alpha"]
            fc_vals = prever_lr(todos_vals, n_lags, alpha_lr_otimo, HORIZONTE)

        elif melhor == "SVR":
            params_svr = ast.literal_eval(resultados_serie["SVR"]["params"])
            fc_vals    = prever_svr(todos_vals, n_lags, params_svr, HORIZONTE)

        else:  # MLP
            params_mlp = ast.literal_eval(resultados_serie["MLP"]["params"])
            fc_vals    = prever_mlp(todos_vals, n_lags, params_mlp, HORIZONTE)

        # Reverte shift (se aplicável)
        fc_originais = (
            reverter_shift(np.array(fc_vals), shift_value).tolist()
            if shift_value > 0 else fc_vals
        )

        if verbose:
            print(f"  Último real {FIM_SERIE} : {_fmt(float(serie_original.iloc[-1]))}")
            for mes, val in zip(MESES_PREV, fc_originais):
                print(f"  Previsão {mes}    : {_fmt(val)}")

        # Métricas de variação (mês a mês)
        ultimo_real = float(serie_original.iloc[-1])
        vars_pct = []
        prev_ant = ultimo_real
        for val in fc_originais:
            var = (
                ((val - prev_ant) / abs(prev_ant) * 100)
                if prev_ant != 0 else float("nan")
            )
            vars_pct.append(var)
            prev_ant = val

        # Armazena resultados
        previsoes[codigo] = {
            "nome"             : nome,
            "grupo_sinal"      : grupo_sinal,
            "shift_value"      : shift_value,
            "melhor_modelo"    : melhor,
            "params"           : resultados_serie[melhor]["params"],
            "RMSE_teste"       : resultados_serie[melhor]["RMSE"],
            "MAE_teste"        : resultados_serie[melhor]["MAE"],
            "MAPE_teste"       : resultados_serie[melhor]["MAPE"],
            "fc_originais"     : fc_originais,
            "resultados_todos" : resultados_serie,
            "treino_vals"      : treino_vals,
            "teste_vals"       : teste_vals,
            "treino_idx"       : treino_idx,
            "teste_idx"        : teste_idx,
            "todos_vals"       : todos_vals,
            "periodos"         : serie_pronta.index,
            "n_holdout"        : n_holdout,
            "n_lags"           : n_lags,
            "n_treino"         : n_treino,
        }

        # Registros para CSV de métricas
        for mod, res in resultados_serie.items():
            registros_met.append({
                "timestamp"     : timestamp,
                "Codigo"        : codigo,
                "Nome"          : nome,
                "Grupo_Sinal"   : grupo_sinal,
                "N_Total"       : len(serie_original),
                "N_Treino"      : n_treino,
                "N_Holdout"     : n_holdout,
                "N_Lags"        : n_lags,
                "Modelo"        : mod,
                "Parametros"    : res["params"],
                "RMSE"          : res["RMSE"],
                "MAE"           : res["MAE"],
                "MAPE"          : res["MAPE"],
                "Selecionado"   : mod == melhor,
                "Ruido_Branco"  : False,
                "LB_Pvalor"     : acf_info["lb_pvalor"],
                "Lags_Sig_PACF" : str(acf_info["lags_sig_pacf"]),
                "Metodo_Lags"   : acf_info["metodo_lags"],
            })

        # Registros para CSV de parâmetros
        registros_params.append({
            "timestamp"    : timestamp,
            "Codigo"       : codigo,
            "Nome"         : nome,
            "Grupo_Sinal"  : grupo_sinal,
            "Shift_Value"  : shift_value,
            "Melhor_Modelo": melhor,
            "Params_ARIMA" : resultados_serie.get("ARIMA", {}).get("params", "N/A"),
            "Params_LR"    : resultados_serie.get("LR",    {}).get("params", "N/A"),
            "Params_SVR"   : resultados_serie.get("SVR",   {}).get("params", "N/A"),
            "Params_MLP"   : resultados_serie.get("MLP",   {}).get("params", "N/A"),
        })

        # Registros para CSV de previsões
        fim_col = FIM_SERIE.replace("-", "_")
        registro_prev = {
            "timestamp"              : timestamp,
            "Codigo"                 : codigo,
            "Nome"                   : nome,
            "Grupo_Sinal"            : grupo_sinal,
            "Melhor_Modelo"          : melhor,
            "Parametros"             : resultados_serie[melhor]["params"],
            "RMSE_Teste"             : round(resultados_serie[melhor]["RMSE"], 2),
            "MAE_Teste"              : round(resultados_serie[melhor]["MAE"],  2),
            "MAPE_Teste_Pct"         : round(resultados_serie[melhor]["MAPE"], 4),
            "LB_Pvalor"              : round(acf_info["lb_pvalor"], 4) if not np.isnan(acf_info["lb_pvalor"]) else np.nan,
            "Lags_Sig_PACF"          : str(acf_info["lags_sig_pacf"]),
            "Metodo_Lags"            : acf_info["metodo_lags"],
            "N_Lags"                 : n_lags,
            f"Ultimo_Real_{fim_col}" : round(ultimo_real, 2),
        }
        for mes, val, var in zip(MESES_PREV, fc_originais, vars_pct):
            mes_col = mes.replace("-", "_")
            registro_prev[f"Prev_{mes_col}"]    = round(val, 2)
            registro_prev[f"Var_{mes_col}_pct"] = round(var, 2) if not np.isnan(var) else np.nan
        registros_prev.append(registro_prev)

        # Checkpoint
        concluidos.add(str(codigo))
        _salvar_checkpoint(registros_met, registros_params, registros_prev, timestamp)

        # Gráfico
        if gerar_graficos:
            caminho_plot = plotar_serie(
                codigo=codigo,
                nome=nome,
                grupo_sinal=grupo_sinal,
                treino_idx=treino_idx,
                treino_vals=treino_vals,
                teste_idx=teste_idx,
                teste_vals=teste_vals,
                resultados_serie=resultados_serie,
                melhor_modelo=melhor,
                meses_prev=meses_prev,
                fc_vals=fc_originais,
                n_holdout=n_holdout,
                n_lags=n_lags,
                mostrar=False,
            )
            logger.info("[%s] Gráfico salvo: %s", codigo, caminho_plot.name)

    # 3. Salva CSVs
    metricas_df = pd.DataFrame(registros_met)
    params_df   = pd.DataFrame(registros_params)
    prev_df     = pd.DataFrame(registros_prev)

    if salvar_resultados:
        metricas_df.to_csv(RESULTS_DIR / f"metricas_{timestamp}.csv",    index=False)
        params_df.to_csv(  RESULTS_DIR / f"parametros_{timestamp}.csv",  index=False)
        prev_df.to_csv(    RESULTS_DIR / f"previsoes_{timestamp}.csv",    index=False)
        logger.info("CSVs salvos em results/ com timestamp %s", timestamp)

        excel_paths = gerar_excel_resultados(metricas_df, prev_df, timestamp)
        logger.info("Excel diagnóstico : %s", excel_paths["diagnostico"].name)
        logger.info("Excel previsões   : %s", excel_paths["previsoes"].name)

    # Apaga checkpoint ao concluir com sucesso
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint removido após conclusão bem-sucedida.")

    logger.info("Experimento concluído. %d séries processadas.", len(previsoes))

    return {
        "previsoes"  : previsoes,
        "metricas_df": metricas_df,
        "params_df"  : params_df,
        "prev_df"    : prev_df,
        "timestamp"  : timestamp,
        "excel_paths": excel_paths if salvar_resultados else {},
    }


# Entry point
if __name__ == "__main__":
    resultados = rodar_experimento(
        salvar_resultados=True,
        gerar_graficos=True,
        verbose=True,
    )
    print("\n-- Previsões Finais ----------------------------------------------")
    colunas_prev = ["Codigo", "Melhor_Modelo", "RMSE_Teste"] + [
        f"Prev_{m.replace('-', '_')}" for m in MESES_PREV[:2]
    ]
    colunas_prev = [c for c in colunas_prev if c in resultados["prev_df"].columns]
    print(resultados["prev_df"][colunas_prev].to_string(index=False))
