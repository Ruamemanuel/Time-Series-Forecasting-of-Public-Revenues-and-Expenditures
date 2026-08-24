"""
Experimento dos modelos globais: cada modelo é treinado uma vez sobre todas as
séries não-ruído-branco e avaliado de forma recursiva.

Modelos: Ridge (efeitos fixos por série, via one-hot do series_id), Random Forest
e MLP (series_id numérico) e LightGBM (series_id categórico nativo, a referência).
Todos partilham o mesmo dataset (lags + calendário + series_id), a normalização
por série e a previsão recursiva.

Ordem para não vazar: divide treino/conjunto de teste, identifica as séries modeláveis pelo
Ljung-Box no treino, monta o dataset global uma vez com os treinos, treina cada
modelo, avalia recursivamente no conjunto de teste e compara com os resultados individuais
do experiment.py. No fim, exporta CSV + Excel.

Rodar com: python experiment_global.py
"""

from __future__ import annotations

import glob
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CAMINHO_PARQUET, RESULTS_DIR,
    N_LAGS_GLOBAL, RANDOM_STATE, MESES_PREV,
)
from utils import (
    carregar_dados, construir_series, mapear_nomes,
    filtrar_series, preparar_serie, dividir_serie,
)
from utils.acf_analysis import analisar_serie
from utils.metrics import calcular_metricas
from models.lgbm_global import (
    construir_dataset_global,
    treinar_lgbm_global,
    prever_rolling_lgbm,
    avaliar_lgbm,
    _col_names,
)
from models.global_sklearn import (
    treinar_global_ridge,
    treinar_global_rf,
    treinar_global_mlp,
    prever_rolling_global_sklearn,
)
from reports.plots import plotar_serie

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment_global")

# Checkpoint
# Versao do schema — altere para invalidar checkpoints antigos ao mudar o protocolo.
_CHECKPOINT_VERSION = "v1_global_multistep_v2"  # v2: grids unificados (RF/MLP GridSearch, LGBM params search)
_CHECKPOINT_PATH    = RESULTS_DIR / "checkpoint_global.pkl"


def _salvar_checkpoint(registros: list, timestamp: str) -> None:
    """Persiste registros concluidos em disco."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "registros"     : registros,
        "timestamp"     : timestamp,
        "concluidos"    : {str(r["Codigo"]) for r in registros},
        "schema_version": _CHECKPOINT_VERSION,
    }
    with open(_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(data, f)


def _carregar_checkpoint() -> dict | None:
    """
    Carrega checkpoint existente, validando versao do schema.
    Retorna None se inexistente ou schema diferente.
    """
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
        n = len(data.get("registros", []))
        logger.info(
            "Checkpoint encontrado: %d series ja avaliadas (schema=%s). Retomando...",
            n, versao,
        )
        return data
    except Exception as exc:
        logger.warning("Falha ao carregar checkpoint: %s. Iniciando do zero.", exc)
        return None


# Helpers

def _fmt(v: float) -> str:
    if pd.isna(v):    return "N/A"
    if abs(v) >= 1e9: return f"R$ {v/1e9:.2f} Bi"
    if abs(v) >= 1e6: return f"R$ {v/1e6:.2f} Mi"
    if abs(v) >= 1e3: return f"R$ {v/1e3:.2f} K"
    return f"R$ {v:,.2f}"


def _carregar_resultados_individuais() -> pd.DataFrame | None:
    csvs = sorted(glob.glob(str(RESULTS_DIR / "metricas_*.csv")))
    csvs = [c for c in csvs
            if "comparacao_local" not in Path(c).name
            and "comparacao_lgbm"  not in Path(c).name
            and "comparacao_global" not in Path(c).name]
    if not csvs:
        logger.warning("Nenhum CSV de metricas individuais. Execute experiment.py primeiro.")
        return None
    path = csvs[-1]
    logger.info("Resultados individuais: %s", Path(path).name)
    return pd.read_csv(path)


def _wilcoxon_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = (a - b)[~np.isnan(a - b)]
    if len(diff) < 5:
        return float("nan"), float("nan")
    try:
        s, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


# Orquestrador principal

def rodar_experimento_global(
    caminho_parquet : Path = CAMINHO_PARQUET,
    n_lags          : int  = N_LAGS_GLOBAL,
    random_state    : int  = RANDOM_STATE,
    verbose         : bool = True,
    gerar_graficos  : bool = True,
) -> dict:
    """
    Treina e avalia Ridge, RF, MLP e LGBM globais. Compara com individual.

    Suporta retomada via checkpoint: se o script for interrompido durante a
    avaliacao por serie, recarrega o progresso salvo e pula as series ja
    concluidas (os modelos sao re-treinados, pois nao sao serializados).

    Returns
    -------
    dict com comparacao_df, modelos, norm_params, label_enc, timestamp
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tenta retomar checkpoint
    ckpt = _carregar_checkpoint()
    if ckpt is not None:
        timestamp   = ckpt["timestamp"]
        registros   = ckpt["registros"]
        concluidos  = ckpt["concluidos"]
        n_retomados = len(registros)
        logger.info("Retomando: %d series ja avaliadas.", n_retomados)
    else:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        registros  = []
        concluidos = set()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carrega e divide séries
    logger.info("Carregando dados de %s", caminho_parquet)
    df         = carregar_dados(caminho_parquet)
    series_raw = construir_series(df)
    nomes      = mapear_nomes(df)
    series, _  = filtrar_series(series_raw)
    logger.info("Series validas: %d", len(series))

    treinos    : dict = {}
    testes     : dict = {}
    treino_idx : dict = {}
    teste_idx  : dict = {}
    acf_infos  : dict = {}
    grupos     : dict = {}
    shifts     : dict = {}

    for cod, serie_original in series.items():
        serie_pronta, grupo_sinal, shift_value = preparar_serie(serie_original)
        grupos[cod] = grupo_sinal
        shifts[cod] = shift_value

        (tv, tsv, tidx, tesidx, _, _) = dividir_serie(serie_pronta)
        treinos[cod]   = tv
        testes[cod]    = tsv
        treino_idx[cod] = tidx
        teste_idx[cod]  = tesidx
        acf_infos[cod]  = analisar_serie(tv)

    series_modelaveis = [c for c, info in acf_infos.items()
                         if not info["e_ruido_branco"]]
    logger.info("Series modelaveis: %d / %d", len(series_modelaveis), len(series))

    # 2. Constrói dataset global (UMA VEZ)
    logger.info("Construindo dataset global com %d series...", len(series_modelaveis))
    series_data = {
        cod: (treinos[cod], treino_idx[cod])
        for cod in series_modelaveis
    }
    X_df, y_global, norm_params, label_enc = construir_dataset_global(
        series_data, n_lags=n_lags
    )
    col_names = _col_names(n_lags)
    logger.info("Dataset global: %d obs | %d features", len(X_df), X_df.shape[1])

    # 3. Treina todos os modelos globais
    logger.info("Treinando Ridge global...")
    modelo_ridge, info_ridge = treinar_global_ridge(X_df, y_global, random_state)

    logger.info("Treinando Random Forest global...")
    modelo_rf, info_rf = treinar_global_rf(X_df, y_global, random_state)

    logger.info("Treinando MLP global...")
    modelo_mlp, info_mlp = treinar_global_mlp(X_df, y_global, random_state)

    logger.info("Treinando LGBM global...")
    modelo_lgbm, n_est_lgbm = treinar_lgbm_global(X_df, y_global,
                                                    random_state=random_state)

    modelos = {
        "Ridge": (modelo_ridge, f"alpha={info_ridge['alpha']}"),
        "RF"   : (modelo_rf,    f"n_est={info_rf['n_estimators']}"),
        "MLP"  : (modelo_mlp,   f"hidden={info_mlp['hidden_layer_sizes']} n_iter={info_mlp['n_iter']}"),
        "LGBM" : (modelo_lgbm,  f"n_est={n_est_lgbm}"),
    }

    # 4. Avalia cada modelo por série
    logger.info("Avaliando modelos por serie (%d series modelaveis)...",
                len(series_modelaveis))

    resultados_ind = _carregar_resultados_individuais()
    ind_map = {}
    if resultados_ind is not None:
        for _, row in resultados_ind.iterrows():
            cod = row.get("Codigo") or row.get("codigo")
            if cod is not None:
                ind_map[str(cod)] = row

    n_total   = len(series_modelaveis)

    for i, cod in enumerate(series_modelaveis, 1):
        cod_str   = str(cod)

        # Pula series ja avaliadas (retomada de checkpoint)
        if cod_str in concluidos:
            logger.debug("  [%d/%d] %s — ja concluida, pulando.", i, n_total, cod_str)
            continue
        nome      = nomes.get(cod, cod_str)
        tv        = treinos[cod]
        tsv       = testes[cod]
        tidx      = treino_idx[cod]
        tsidx     = teste_idx[cod]

        if cod_str not in norm_params:
            continue
        mean, std = norm_params[cod_str]

        try:
            sid = int(label_enc.transform([cod_str])[0])
        except Exception:
            continue

        reg : dict = {
            "Codigo" : cod,
            "Nome"   : nome,
            "N_Treino" : len(tv),
            "N_Holdout": len(tsv),
        }

        # Individual (carregado do experiment.py)
        row_ind = ind_map.get(cod_str, {})
        reg["Ind_Melhor_Modelo"] = row_ind.get("Melhor_Modelo", "N/A")
        reg["Ind_RMSE"] = float(row_ind.get("RMSE_Melhor", float("nan")))
        reg["Ind_MAE"]  = float(row_ind.get("MAE_Melhor",  float("nan")))
        reg["Ind_MAPE"] = float(row_ind.get("MAPE_Melhor", float("nan")))

        # Avalia cada modelo global
        for nome_mod, (modelo, params_str) in modelos.items():
            try:
                if nome_mod == "LGBM":
                    preds = prever_rolling_lgbm(
                        modelo, tv, tsv, tidx, tsidx,
                        n_lags, mean, std, sid,
                    )
                elif nome_mod == "RF":
                    preds = prever_rolling_global_sklearn(
                        modelo, tv, tsv, tsidx, n_lags,
                        mean, std, sid, col_names, usa_array=True,
                    )
                else:   # Ridge ou MLP (Pipeline que aceita DataFrame ou array)
                    preds = prever_rolling_global_sklearn(
                        modelo, tv, tsv, tsidx, n_lags,
                        mean, std, sid, col_names, usa_array=False,
                    )
                metricas = calcular_metricas(np.asarray(tsv, dtype=float), preds)
                reg[f"Global_{nome_mod}_RMSE"]   = metricas["RMSE"]
                reg[f"Global_{nome_mod}_MAE"]    = metricas["MAE"]
                reg[f"Global_{nome_mod}_MAPE"]   = metricas["MAPE"]
                reg[f"Global_{nome_mod}_Params"] = params_str
            except Exception as exc:
                logger.warning("[%s] %s falhou: %s", cod, nome_mod, exc)
                reg[f"Global_{nome_mod}_RMSE"]   = float("nan")
                reg[f"Global_{nome_mod}_MAE"]    = float("nan")
                reg[f"Global_{nome_mod}_MAPE"]   = float("nan")
                reg[f"Global_{nome_mod}_Params"] = f"erro: {exc}"

        # Melhor modelo global
        rmses_glob = {
            m: reg.get(f"Global_{m}_RMSE", float("nan"))
            for m in modelos
        }
        rmses_validos = {m: v for m, v in rmses_glob.items()
                         if not np.isnan(v)}
        if rmses_validos:
            melhor_glob  = min(rmses_validos, key=rmses_validos.get)
            reg["Global_Melhor_Modelo"] = melhor_glob
            reg["Global_Melhor_RMSE"]  = rmses_validos[melhor_glob]
        else:
            reg["Global_Melhor_Modelo"] = "N/A"
            reg["Global_Melhor_RMSE"]  = float("nan")

        # Delta vs individual
        ind_rmse = reg["Ind_RMSE"]
        mel_rmse = reg["Global_Melhor_RMSE"]
        reg["Delta_RMSE_pct"] = (
            (mel_rmse - ind_rmse) / ind_rmse * 100.0
            if not (np.isnan(ind_rmse) or np.isnan(mel_rmse) or ind_rmse == 0)
            else float("nan")
        )

        registros.append(reg)
        concluidos.add(cod_str)
        _salvar_checkpoint(registros, timestamp)

        # Gráfico de previsão global (se habilitado)
        if gerar_graficos:
            try:
                melhor_glob = reg.get("Global_Melhor_Modelo", "N/A")
                preds_glob  = reg.get(f"Global_{melhor_glob}_RMSE", float("nan"))
                if melhor_glob != "N/A" and not np.isnan(preds_glob):
                    # Re-obtém previsões do melhor modelo para plotar
                    _modelo_glob, _ = modelos.get(melhor_glob, (None, None))
                    if _modelo_glob is not None:
                        if melhor_glob == "LGBM":
                            _preds_plot = prever_rolling_lgbm(
                                _modelo_glob, tv, tsv, tidx, tsidx,
                                n_lags, mean, std, sid,
                            )
                        elif melhor_glob == "RF":
                            _preds_plot = prever_rolling_global_sklearn(
                                _modelo_glob, tv, tsv, tsidx, n_lags,
                                mean, std, sid, col_names, usa_array=True,
                            )
                        else:
                            _preds_plot = prever_rolling_global_sklearn(
                                _modelo_glob, tv, tsv, tsidx, n_lags,
                                mean, std, sid, col_names, usa_array=False,
                            )
                        _rmse_plot = reg.get(f"Global_{melhor_glob}_RMSE", float("nan"))
                        _mape_plot = reg.get(f"Global_{melhor_glob}_MAPE", float("nan"))
                        _res_plot  = {
                            "Global_" + melhor_glob: {
                                "preds_teste": _preds_plot,
                                "RMSE": _rmse_plot,
                                "MAPE": _mape_plot,
                            }
                        }
                        meses_prev_period = [pd.Period(m, freq="M") for m in MESES_PREV]
                        plotar_serie(
                            codigo      = cod_str,
                            nome        = nome,
                            grupo_sinal = grupos.get(cod, ""),
                            treino_idx  = tidx,
                            treino_vals = tv,
                            teste_idx   = tsidx,
                            teste_vals  = np.asarray(tsv),
                            resultados_serie = _res_plot,
                            melhor_modelo = "Global_" + melhor_glob,
                            meses_prev  = meses_prev_period,
                            fc_vals     = [],
                            n_holdout   = len(tsv),
                            n_lags      = n_lags,
                            mostrar     = False,
                            sufixo      = "_global",
                        )
            except Exception as exc_plot:
                logger.warning("[%s] Erro ao gerar gráfico global: %s", cod, exc_plot)

        if verbose and i % 20 == 0:
            logger.info("  %d / %d series avaliadas", i, n_total)

    # 5. Monta DataFrame
    comp_df = pd.DataFrame(registros)

    # 6. Resumo por modelo
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTADOS GLOBAIS — COMPARACAO COM INDIVIDUAL")
        print(f"{'='*70}")
        print(f"{'Modelo':<12} {'RMSE medio':>14} {'Wins':>8} {'Delta med%':>11} "
              f"{'Wilcoxon p':>12}")
        print("-" * 70)

        ind_rmse_arr = comp_df["Ind_RMSE"].values.astype(float)

        for nome_mod in list(modelos.keys()) + ["Melhor"]:
            col = (f"Global_{nome_mod}_RMSE" if nome_mod != "Melhor"
                   else "Global_Melhor_RMSE")
            if col not in comp_df.columns:
                continue
            g_arr   = comp_df[col].values.astype(float)
            mask    = ~(np.isnan(ind_rmse_arr) | np.isnan(g_arr))
            wins    = int(np.sum(g_arr[mask] < ind_rmse_arr[mask]))
            n_valid = int(mask.sum())
            rmse_med = float(np.nanmean(g_arr))
            delta_arr = (g_arr - ind_rmse_arr) / np.where(ind_rmse_arr == 0, np.nan, ind_rmse_arr) * 100
            delta_med = float(np.nanmedian(delta_arr))
            _, p_wx  = _wilcoxon_safe(ind_rmse_arr[mask], g_arr[mask])
            print(f"  {nome_mod:<10} {rmse_med:>14,.0f} {wins:>5}/{n_valid:<3} "
                  f"{delta_med:>+10.1f}%  p={p_wx:.4f}")

        print(f"\n  Individual (referencia): RMSE medio = "
              f"{float(np.nanmean(ind_rmse_arr)):,.0f}")
        print(f"{'='*70}")

    # 7. Exporta
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = RESULTS_DIR / f"comparacao_global_{timestamp}.csv"
    xlsx_path = RESULTS_DIR / f"comparacao_global_{timestamp}.xlsx"

    comp_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        comp_df.to_excel(writer, sheet_name="Global", index=False)
        ws = writer.sheets["Global"]
        rmse_cols = [c for c in comp_df.columns if "RMSE" in c]
        for col_name in rmse_cols:
            col_idx = list(comp_df.columns).index(col_name) + 1
            col_letter = ""
            n = col_idx
            while n:
                col_letter = chr(64 + (n - 1) % 26 + 1) + col_letter
                n = (n - 1) // 26
            for row in range(2, len(comp_df) + 2):
                ws[f"{col_letter}{row}"].number_format = "#,##0.00"

    logger.info("Resultados salvos: %s | %s", csv_path.name, xlsx_path.name)
    print(f"\nCSV : {csv_path}")
    print(f"XLSX: {xlsx_path}")

    # Remove checkpoint — experimento concluido com sucesso
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint removido.")

    return {
        "comparacao_df" : comp_df,
        "modelos"       : {k: v[0] for k, v in modelos.items()},
        "norm_params"   : norm_params,
        "label_enc"     : label_enc,
        "timestamp"     : timestamp,
    }


if __name__ == "__main__":
    rodar_experimento_global()
