"""
Parâmetros globais do experimento. Tudo que dá para parametrizar mora aqui; os
módulos importam deste arquivo em vez de fixar valores no código.
"""

from pathlib import Path

# Caminhos
ROOT_DIR    = Path(__file__).parent
REPO_ROOT   = ROOT_DIR.parent.parent
DATA_DIR    = REPO_ROOT / "data" / "receita"
RESULTS_DIR = REPO_ROOT / "results" / "receita"
PLOTS_DIR   = RESULTS_DIR / "plots"
REPORTS_DIR = ROOT_DIR / "reports"

CAMINHO_PARQUET = DATA_DIR / "receita_realizada_a_partir_2019_sgo.parquet"

# Filtragem de séries
FIM_SERIE   = "2025-10"   # apenas séries cujo último período seja este
MIN_SERIE   = 10          # comprimento mínimo para incluir a série (obs. totais)
MIN_TREINO  = 15          # mínimo de obs. no treino para habilitar SVR, MLP e LR

# Divisão treino / teste
MAX_HOLDOUT = 12
PCT_HOLDOUT = 0.20
MIN_HOLDOUT = 3

# Janela de lags (LR / SVR / MLP)
MAX_LAGS  = 12
PCT_LAGS  = 0.30
MIN_LAGS  = 3

# Cross-validation
N_SPLITS_CV         = 3
MIN_AMOSTRAS_EARLY  = 20   # fold mínimo para ativar early_stopping no MLP

# Horizonte de previsão
HORIZONTE    = 12
MESES_PREV   = [
    "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
]

# Tratamento de sinal
TRATAMENTO_NEGATIVOS = "shift"

# Regressão Linear
LR_ALPHA = 1.0   # regularização Ridge

# Grids de hiperparâmetros
GRID_SVR = {
    "C"      : [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 1],
    "kernel" : ["rbf", "linear"],
}

GRID_MLP = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "alpha"             : [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.01],
}

# Grids unificados (usados em TODOS os paradigmas)
GRID_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

GRID_RF = {
    "n_estimators"    : [100, 300, 500],
    "max_depth"       : [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
    # "sqrt"  = recomendação original Breiman (2001) para classificação
    # 0.33    = p/3, recomendação de Liaw & Wiener (2002) para regressão
    # "log2"  = alternativa parcimonial para datasets de alta dimensão
    # Probst, Wright & Boulesteix (2019) identificam max_features como o
    # hiperparâmetro de maior impacto no RF — por isso entra na busca.
    "max_features"    : ["sqrt", "log2", 0.33],
}

GRID_LGBM_PARAMS = {
    "num_leaves"       : [15, 31, 63],
    "min_child_samples": [5, 20, 50],
}

# Ruído branco + seleção de lags (detalhes em utils/acf_analysis.py):
# Ljung-Box no treino — se p >= ACF_ALPHA, é ruído branco e a série é pulada.
# Caso contrário, os lags com |PACF| > 1.96/√n são candidatos e
# n_lags = min(maior lag significativo, MAX_LAGS). Se o Ljung-Box rejeita mas o
# PACF não acusa nada, usa o fallback min(MAX_LAGS, max(MIN_LAGS, n_treino*PCT_LAGS)).

ACF_ALPHA        = 0.05   # nível de significância — bandas PACF e Ljung-Box
LB_NLAGS_RATIO   = 0.20   # fração do treino para definir h no Ljung-Box
LB_NLAGS_MIN     = 5      # mínimo de lags no Ljung-Box
LB_NLAGS_MAX     = 20     # máximo de lags no Ljung-Box

# Ablation study
ABLATION_LAGS     = [3, 6, 12]          # configurações de lags a comparar
ABLATION_HOLDOUTS = [3, 6, 12]          # tamanhos de conjunto de teste a comparar

# LightGBM global
# Janela de lags FIXA para o modelo global (feature space uniforme entre séries)
N_LAGS_GLOBAL           = 12

# Hiperparâmetros: baseados em literatura de competições de séries temporais
# (M5 Competition, Ke et al. 2017) e validados por early stopping interno.
LGBM_N_ESTIMATORS       = 1000   # máximo; early stopping determina o ótimo
LGBM_LEARNING_RATE      = 0.05
LGBM_NUM_LEAVES         = 31
LGBM_MIN_CHILD_SAMPLES  = 20     # regularização — evita overfit em séries curtas
LGBM_SUBSAMPLE          = 0.8
LGBM_COLSAMPLE_BYTREE   = 0.8
LGBM_REG_ALPHA          = 0.1    # L1
LGBM_REG_LAMBDA         = 0.1    # L2
ANO_BASE_LGBM           = 2019   # year_offset = year - ANO_BASE (proxy de tendência)

# Reprodutibilidade
RANDOM_STATE = 42

# Paleta visual
CORES = {
    "azul_esc" : "#1B3A5C",
    "azul_med" : "#2E6DA4",
    "azul_cla" : "#D6E8F7",
    "verde"    : "#27AE60",
    "vermelho" : "#C0392B",
    "amarelo"  : "#F39C12",
    "cinza_cla": "#F4F6F9",
    "cinza_med": "#BDC3C7",
    "branco"   : "#FFFFFF",
    "roxo"     : "#8E44AD",
    "laranja"  : "#E67E22",
}

# Cor por modelo — usado em gráficos
COR_MODELO = {
    "ARIMA": "#C0392B",
    "SVR"  : "#F39C12",
    "MLP"  : "#8E44AD",
    "LR"   : "#E67E22",
}
