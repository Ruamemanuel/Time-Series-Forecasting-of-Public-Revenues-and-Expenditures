"""
Carregamento dos dados de receita, filtragem das séries e tratamento de sinal.

Quanto ao sinal, cada série cai num grupo:
  apenas_positivos      sem transformação
  apenas_negativos      shift de |min| + 1 (reversível)
  positivos_e_negativos sem transformação (os negativos são estornos/ajustes)
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CAMINHO_PARQUET, FIM_SERIE, MIN_SERIE,
    MAX_HOLDOUT, PCT_HOLDOUT, MIN_HOLDOUT,
    MAX_LAGS, PCT_LAGS, MIN_LAGS, MIN_TREINO,
)

logger = logging.getLogger(__name__)


# Carregamento

def carregar_dados(caminho: Path = CAMINHO_PARQUET) -> pd.DataFrame:
    """Lê o parquet e cria a coluna 'Periodo' (Period[M], a partir de Ano/Mês)."""
    df = pd.read_parquet(caminho)
    df["Periodo"] = pd.to_datetime(
        df["Ano"].astype(str) + "-" + df["Mês"].astype(str).str.zfill(2)
    ).dt.to_period("M")
    logger.info("Dados carregados: %d linhas, %d colunas", *df.shape)
    return df


def construir_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Soma os valores por (código de fonte, período) e devolve {codigo: série}
    com índice mensal."""
    agrupado = (
        df.groupby(["Fonte_Det_Cód_Harm", "Periodo"])["Valor_ReceitaRealizada"]
        .sum()
        .reset_index()
        .sort_values(["Fonte_Det_Cód_Harm", "Periodo"])
    )
    series = {}
    for cod, grupo in agrupado.groupby("Fonte_Det_Cód_Harm"):
        s = grupo.set_index("Periodo")["Valor_ReceitaRealizada"].sort_index()
        series[cod] = s
    logger.info("Séries construídas: %d códigos únicos", len(series))
    return series


def mapear_nomes(df: pd.DataFrame) -> dict:
    """Retorna {Fonte_Det_Cód_Harm: Fonte_Det_Nome_Harm}."""
    return (
        df[["Fonte_Det_Cód_Harm", "Fonte_Det_Nome_Harm"]]
        .drop_duplicates("Fonte_Det_Cód_Harm")
        .set_index("Fonte_Det_Cód_Harm")["Fonte_Det_Nome_Harm"]
        .to_dict()
    )


# Filtragem

def filtrar_series(
    series: dict[str, pd.Series],
    fim: str = FIM_SERIE,
    min_obs: int = MIN_SERIE,
) -> tuple[dict, list]:
    """Mantém só as séries que terminam em `fim` e têm ao menos `min_obs` pontos.
    Devolve (series_validas, excluidas), onde excluidas é uma lista de
    (codigo, motivo)."""
    fim_period = pd.Period(fim, freq="M")
    validas, excluidas = {}, []

    for cod, serie in series.items():
        ultimo = serie.index.max()
        if ultimo != fim_period:
            excluidas.append((cod, f"ultimo periodo {ultimo} != {fim_period}"))
            continue
        if len(serie) < min_obs:
            excluidas.append((cod, f"série curta ({len(serie)} obs. < {min_obs})"))
            continue
        validas[cod] = serie

    logger.info(
        "Filtragem: %d séries válidas, %d excluídas",
        len(validas), len(excluidas),
    )
    return validas, excluidas


# Classificação de sinal

def classificar_sinal(serie: pd.Series) -> str:
    """Classifica a série em apenas_positivos, apenas_negativos ou
    positivos_e_negativos."""
    tem_pos = (serie > 0).any()
    tem_neg = (serie < 0).any()
    if tem_pos and not tem_neg:
        return "apenas_positivos"
    if tem_neg and not tem_pos:
        return "apenas_negativos"
    return "positivos_e_negativos"


# Tratamento de sinal

def aplicar_shift(serie: pd.Series) -> tuple[pd.Series, float]:
    """Desloca a série por |min| + 1 para deixar tudo positivo. Devolve
    (serie_transformada, shift_value)."""
    shift_value = abs(serie.min()) + 1.0
    return serie + shift_value, shift_value


def reverter_shift(valores: np.ndarray, shift_value: float) -> np.ndarray:
    """Reverte o shift aplicado pela função `aplicar_shift`."""
    return valores - shift_value


def preparar_serie(serie: pd.Series) -> tuple[pd.Series, str, float]:
    """Aplica o tratamento conforme o grupo de sinal. Devolve (serie_pronta,
    grupo_sinal, shift_value), com shift_value = 0.0 quando não há transformação."""
    grupo = classificar_sinal(serie)
    if grupo == "apenas_negativos":
        serie_pronta, shift_value = aplicar_shift(serie)
        logger.debug("Série %s: shift aplicado (%.2f)", serie.name, shift_value)
    else:
        serie_pronta = serie.copy()
        shift_value  = 0.0
    return serie_pronta, grupo, shift_value


# Parâmetros adaptativos

def calcular_holdout(n_total: int) -> int:
    """N_HOLDOUT = min(MAX_HOLDOUT, 20% do total), mínimo MIN_HOLDOUT."""
    return max(MIN_HOLDOUT, min(MAX_HOLDOUT, int(n_total * PCT_HOLDOUT)))


def calcular_lags(n_treino: int) -> int:
    """N_LAGS = min(MAX_LAGS, 30% do treino), mínimo MIN_LAGS."""
    return max(MIN_LAGS, min(MAX_LAGS, int(n_treino * PCT_LAGS)))


def dividir_serie(
    serie: pd.Series,
) -> tuple[np.ndarray, np.ndarray, object, object, int, int]:
    """Separa treino e conjunto de teste. Devolve treino_vals, teste_vals, treino_idx,
    teste_idx, n_holdout e n_lags."""
    valores   = serie.values.astype(float)
    periodos  = serie.index
    n_holdout = calcular_holdout(len(valores))
    n_lags    = calcular_lags(len(valores) - n_holdout)

    treino_vals = valores[:-n_holdout]
    teste_vals  = valores[-n_holdout:]
    treino_idx  = periodos[:-n_holdout]
    teste_idx   = periodos[-n_holdout:]

    return treino_vals, teste_vals, treino_idx, teste_idx, n_holdout, n_lags
