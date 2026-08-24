"""
testes_estatisticos.py
──────────────────────
Testes estatísticos para comparação de modelos de previsão.

Protocolo (Demšar, 2006 — padrão na literatura de ML):
  1. Teste de Friedman  — hipótese global: todos os modelos são equivalentes
  2. Nemenyi post-hoc  — comparações par a par controlando FWER (via scikit_posthocs)
  3. Teste de Wilcoxon  — comparações par a par (signed-rank, two-sided)
  4. Correção de Holm-Bonferroni  — controla FWER nas k(k-1)/2 comparações
  5. Tamanho de efeito  — r = |Z| / sqrt(n), escala: 0.1=pequeno, 0.3=médio, 0.5=grande

Métrica: nRMSE = RMSE / mean(|y_treino|) — normalizado pela magnitude de cada série.

Três blocos de análise por experimento:
  A. Modelos individuais (ARIMA, LR, SVR, MLP)
  B. Paradigmas — melhor de cada (Ind vs Local vs Global)
  C. Todos os modelos simultâneos
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import scikit_posthocs as sp

warnings.filterwarnings("ignore")

# ── Caminhos ──────────────────────────────────────────────────────────────────
# Raiz do repositório = pasta pai de analysis/
BASE = Path(__file__).resolve().parent.parent

RECEITA = {
    "metricas"   : BASE / "results/receita/metricas_20260527_095838.csv",
    "unificada"  : BASE / "results/receita/comparacao_unificada_20260615_170849.csv",
    "parquet"    : BASE / "data/receita/receita_realizada_a_partir_2019_sgo.parquet",
    "col_valor"  : "Valor_ReceitaRealizada",
    "label"      : "Receita (n=133)",
    "slug"       : "receita",
}

DESPESA = {
    "metricas"   : BASE / "results/despesa/metricas_20260525_121025.csv",
    "unificada"  : BASE / "results/despesa/comparacao_unificada_20260615_170847.csv",
    "parquet"    : BASE / "data/despesa/ts_liquidado.parquet",
    "col_valor"  : "Valor_Liquidado",
    "label"      : "Despesas (n=16)",
    "slug"       : "despesa",
    # 3 séries curtas (N_treino-N_lags<15): Ridge/SVR/MLP individuais só rodaram
    # com hiperparâmetros fixos. Excluídas para que todos os modelos prevejam
    # todas as séries sob o mesmo protocolo (validação cruzada).
    "excluir"    : {501999001, 700005391, 759540000},
}

ALPHA      = 0.05
PLOTS_DIR  = BASE / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Normalização por série ─────────────────────────────────────────────────────

def calcular_normas(parquet_path: Path, col_valor: str, unificada_path: Path) -> dict[int, float]:
    """
    Para cada série, calcula mean(|y_treino|) após eventual shift de sinal.
    Retorna {Codigo: fator_normalizacao}.
    O fator é usado para dividir o RMSE → nRMSE adimensional.
    """
    df_raw = pd.read_parquet(parquet_path)
    # coluna de mês pode ter encoding diferente entre parquets; busca por aproximação
    mes_candidates = [c for c in df_raw.columns if c.lower().replace("\xe3", "a").startswith("m")]
    mes_col = mes_candidates[0] if mes_candidates else "Mês"
    # remove mês=13 (ajustes de fechamento presentes em alguns parquets)
    df_raw = df_raw[df_raw[mes_col].astype(int).between(1, 12)].copy()
    df_raw["Periodo"] = pd.to_datetime(
        df_raw["Ano"].astype(str) + "-" + df_raw[mes_col].astype(int).astype(str).str.zfill(2)
    ).dt.to_period("M")

    agrupado = (
        df_raw.groupby(["Fonte_Det_Cód_Harm", "Periodo"])[col_valor]
        .sum()
        .reset_index()
        .sort_values(["Fonte_Det_Cód_Harm", "Periodo"])
    )

    uni = pd.read_csv(unificada_path)[["Codigo", "N_Holdout"]].drop_duplicates("Codigo")
    n_holdout_map = dict(zip(uni["Codigo"], uni["N_Holdout"]))

    normas = {}
    for cod, grupo in agrupado.groupby("Fonte_Det_Cód_Harm"):
        if cod not in n_holdout_map:
            continue
        valores = grupo["Periodo"].sort_values()  # just to be safe
        vals = grupo.sort_values("Periodo")[col_valor].values.astype(float)
        n_holdout = int(n_holdout_map[cod])
        treino = vals[:-n_holdout] if n_holdout > 0 else vals

        # shift se houver negativos (espelha preprocessing.py)
        if treino.min() < 0:
            treino = treino + abs(treino.min()) + 1.0

        fator = np.mean(np.abs(treino))
        normas[cod] = fator if fator > 0 else 1.0  # fallback anti-divisão por zero

    return normas


def normalizar_df(df: pd.DataFrame, rmse_cols: list[str], normas: dict) -> pd.DataFrame:
    """Divide cada coluna de RMSE pelo fator de normalização da série."""
    df = df.copy()
    for col in rmse_cols:
        if col in df.columns:
            df[col] = df.apply(
                lambda row: row[col] / normas.get(int(row["Codigo"]), 1.0), axis=1
            )
    return df


# ── Utilitários estatísticos ──────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def wilcoxon_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    try:
        stat, _ = stats.wilcoxon(x, y, zero_method="wilcox")
        mu    = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z     = (stat - mu) / sigma
        return abs(z) / np.sqrt(n)
    except Exception:
        return np.nan


def interpretar_efeito(r: float) -> str:
    if np.isnan(r):    return "n/a"
    if r < 0.1:        return "negligível"
    if r < 0.3:        return "pequeno"
    if r < 0.5:        return "médio"
    return "grande"


def separador(titulo: str, char: str = "═", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {titulo}")
    print(f"{char * width}")


def sub_separador(titulo: str):
    print(f"\n  {'─' * 76}")
    print(f"  {titulo}")
    print(f"  {'─' * 76}")


# ── Friedman ──────────────────────────────────────────────────────────────────

def friedman_test(nrmse_matrix: pd.DataFrame, label: str) -> tuple[float, float]:
    clean = nrmse_matrix.dropna()
    n_series, n_models = len(clean), len(clean.columns)
    if n_series < 3 or n_models < 2:
        print(f"    [SKIP] amostras insuficientes: n={n_series}, k={n_models}")
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[clean[c].values for c in clean.columns])
    sig = "✓ REJEITA H0" if p < ALPHA else "✗ não rejeita H0"
    print(f"\n  Friedman — {label}")
    print(f"    n={n_series} séries  |  k={n_models} modelos")
    print(f"    χ²({n_models-1}) = {stat:.4f}   p = {p:.6f}   [{sig}]")
    return stat, p


# ── Nemenyi post-hoc ──────────────────────────────────────────────────────────

def nemenyi_posthoc(nrmse_matrix: pd.DataFrame, label: str, indent: int = 4) -> pd.DataFrame:
    """
    Teste post-hoc de Nemenyi após Friedman (via scikit_posthocs).
    Exibe matriz de p-valores e indica quais pares são significativos.
    """
    clean = nrmse_matrix.dropna()
    pad   = " " * indent

    print(f"\n{pad}Nemenyi post-hoc — {label}  (n={len(clean)})")

    # scikit_posthocs espera linhas = observações, colunas = grupos
    pmat = sp.posthoc_nemenyi_friedman(clean.values)
    pmat.index   = clean.columns
    pmat.columns = clean.columns

    cols = clean.columns.tolist()
    print(f"\n{pad}{'Comparação':<50} {'p-Nemenyi':>12}  {'sig?':>5}")
    print(f"{pad}{'-'*70}")
    medians = {c: float(np.median(clean[c])) for c in cols}
    for a, b in itertools.combinations(cols, 2):
        p = pmat.loc[a, b]
        sig = "✓" if p < ALPHA else " "
        melhor = a if medians[a] < medians[b] else b
        print(f"{pad}{a}  vs  {b:<35} {p:>12.5f}  {sig:>5}   {melhor}")

    sig_count = sum(
        1 for a, b in itertools.combinations(cols, 2) if pmat.loc[a, b] < ALPHA
    )
    print(f"\n{pad}Pares significativos (Nemenyi, α={ALPHA}): {sig_count} / {len(list(itertools.combinations(cols, 2)))}")
    return pmat


# ── Wilcoxon pareado ──────────────────────────────────────────────────────────

def wilcoxon_pairwise(
    nrmse_matrix: pd.DataFrame,
    label: str,
    indent: int = 4,
) -> pd.DataFrame:
    clean   = nrmse_matrix.dropna()
    n       = len(clean)
    cols    = clean.columns.tolist()
    pairs   = list(itertools.combinations(cols, 2))
    medians = {c: float(np.median(clean[c])) for c in cols}

    rows  = []
    raw_p = []
    for a, b in pairs:
        try:
            stat, p = stats.wilcoxon(clean[a].values, clean[b].values, zero_method="wilcox")
        except Exception:
            p, stat = np.nan, np.nan
        r = wilcoxon_effect_size(clean[a].values, clean[b].values)
        rows.append({"A": a, "B": b, "p_raw": p, "r": r, "n": n})
        raw_p.append(p if not np.isnan(p) else 1.0)

    reject = holm_bonferroni(raw_p)

    result_rows = []
    for i, row in enumerate(rows):
        row["reject_H0"] = reject[i]
        row["melhor"]    = row["A"] if medians[row["A"]] < medians[row["B"]] else row["B"]
        result_rows.append(row)

    pad = " " * indent
    print(f"\n{pad}Wilcoxon signed-rank + Holm-Bonferroni — {label}  (n={n})")
    print(f"{pad}Mediana nRMSE por modelo:")
    for c, m in sorted(medians.items(), key=lambda x: x[1]):
        print(f"{pad}  {c:<30} {m:>12.6f}")

    print(f"\n{pad}{'Comparação':<50} {'p-raw':>10} {'r':>6} {'efeito':<12} {'sig?':>5} {'melhor'}")
    print(f"{pad}{'-'*100}")
    for i, row in enumerate(result_rows):
        par   = f"{row['A']}  vs  {row['B']}"
        sig   = "✓" if row["reject_H0"] else " "
        ef    = interpretar_efeito(row["r"])
        p_str = f"{row['p_raw']:.5f}" if not np.isnan(row["p_raw"]) else "   nan"
        r_str = f"{row['r']:.3f}"     if not np.isnan(row["r"])     else "  nan"
        print(f"{pad}{par:<50} {p_str:>10} {r_str:>6} {ef:<12} {sig:>5}   {row['melhor']}")

    sig_count = sum(row["reject_H0"] for row in result_rows)
    print(f"\n{pad}Pares significativos após Holm: {sig_count} / {len(result_rows)}")
    return pd.DataFrame(result_rows)


# ── CD Diagram (Demšar 2006) ──────────────────────────────────────────────────

# Valores críticos de Nemenyi (tabela Studentized range / sqrt(2)), α=0.05
_CD_CRITICAL = {
    2: 1.960, 3: 2.344, 4: 2.569,  5: 2.728,  6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
    12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}


def _critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """CD = q_α * sqrt(k(k+1) / 6n)  — fórmula de Demšar (2006)."""
    q = _CD_CRITICAL.get(k, _CD_CRITICAL[max(_CD_CRITICAL)])
    return q * np.sqrt(k * (k + 1) / (6 * n))


def cd_diagram(nrmse_matrix: pd.DataFrame, label: str, slug: str):
    """
    Gera o diagrama de diferença crítica (Demšar 2006).

    Layout padrão:
    - Eixo horizontal no topo: rank 1 (melhor) à esquerda, k (pior) à direita.
    - Primeira metade dos modelos (por rank) → rótulos à ESQUERDA do eixo.
    - Segunda metade → rótulos à DIREITA.
    - Barras azuis ACIMA do eixo conectam grupos sem diferença significativa
      (intervalo contínuo: todos os modelos dentro de CD entre si).
    - Seta vermelha no canto superior esquerdo mostra a CD.
    """
    clean = nrmse_matrix.dropna()
    n, k  = len(clean), len(clean.columns)
    if n < 3 or k < 2:
        print(f"  [CD diagram SKIP] n={n}, k={k}")
        return

    ranks     = clean.rank(axis=1, method="average", ascending=True)
    avg_ranks = ranks.mean().sort_values()   # ordenado do melhor (menor rank) ao pior
    cd        = _critical_difference(k, n)

    models = avg_ranks.index.tolist()        # já ordenados por ranking médio crescente
    r_vals = avg_ranks.values

    # ── grupos de não-significância ───────────────────────────────────────
    # Um grupo é um intervalo contínuo [i, j] tal que r[j] - r[i] <= CD.
    # Varre todos os pares e mantém apenas grupos com ≥ 2 elementos.
    groups = []
    for i in range(k):
        for j in range(i + 1, k):
            if r_vals[j] - r_vals[i] <= cd:
                # verifica se todos os modelos entre i e j estão dentro de CD
                span = r_vals[j] - r_vals[i]
                if span <= cd:
                    groups.append((i, j))

    # colapsa grupos sobrepostos: mantém apenas os maximais
    maximal = []
    for (i, j) in groups:
        dominated = any(
            (a <= i and b >= j and (a, b) != (i, j))
            for (a, b) in groups
        )
        if not dominated:
            maximal.append((i, j))

    # remove grupos de tamanho 1 (modelo vs si mesmo)
    maximal = [(i, j) for (i, j) in maximal if i != j]

    # ── dimensões da figura ───────────────────────────────────────────────
    n_left  = k // 2
    n_right = k - n_left
    row_h   = 0.50                             # altura por linha de rótulo
    n_rows  = max(n_left, n_right)

    # espaço vertical abaixo do eixo: barras de CD + seta CD
    below_h = 0.45 + max(len(maximal), 1) * 0.25

    # eixo fica na base; rótulos crescem para cima
    axis_y  = below_h
    top_label_y = axis_y + n_rows * row_h      # y do rótulo mais alto
    title_h = 0.90                             # espaço reservado para o título
    fig_h   = top_label_y + title_h

    fig, ax = plt.subplots(figsize=(10, max(3.5, fig_h)))
    ax.set_xlim(0, k + 1)
    ax.set_ylim(-0.1, fig_h)
    ax.axis("off")

    # ── eixo de ranking ───────────────────────────────────────────────────
    ax_x0, ax_x1 = 1.0, float(k)        # rank 1 à esquerda, k à direita
    ax.hlines(axis_y, ax_x0, ax_x1, colors="black", linewidths=2)
    for ri in range(1, k + 1):
        ax.vlines(float(ri), axis_y - 0.08, axis_y + 0.08, colors="black", linewidths=1.2)
        ax.text(float(ri), axis_y - 0.22, str(ri), ha="center", va="top", fontsize=8)

    # ── rótulos: primeira metade à esquerda, segunda à direita ───────────
    left_models  = models[:n_left]
    right_models = models[n_left:]

    for idx, m in enumerate(left_models):
        rx   = float(avg_ranks[m])
        label_y = axis_y + (idx + 1) * row_h - 0.1
        # linha vertical do eixo até label_y, depois horizontal até margem
        ax.plot([rx, rx],          [axis_y, label_y], color="dimgray", lw=0.9)
        ax.plot([rx, ax_x0 - 0.1], [label_y, label_y], color="dimgray", lw=0.9)
        ax.text(ax_x0 - 0.18, label_y,
                f"{m} ({avg_ranks[m]:.2f})", ha="right", va="center", fontsize=8.5)

    for idx, m in enumerate(right_models):
        rx      = float(avg_ranks[m])
        label_y = axis_y + (idx + 1) * row_h - 0.1
        ax.plot([rx, rx],          [axis_y, label_y], color="dimgray", lw=0.9)
        ax.plot([rx, ax_x1 + 0.1], [label_y, label_y], color="dimgray", lw=0.9)
        ax.text(ax_x1 + 0.18, label_y,
                f"({avg_ranks[m]:.2f}) {m}", ha="left", va="center", fontsize=8.5)

    # ── barras de não-significância (abaixo do eixo) ─────────────────────
    bar_base = axis_y - 0.22
    bar_gap  = 0.20
    drawn    = []
    for level, (i, j) in enumerate(maximal):
        r_min = float(r_vals[i])
        r_max = float(r_vals[j])
        # empilha verticalmente quando barras se sobrepõem horizontalmente
        by = bar_base
        for prev_by, prev_min, prev_max in drawn:
            if r_min < prev_max + 0.01 and r_max > prev_min - 0.01:
                by = min(by, prev_by - bar_gap)
        ax.hlines(by, r_min, r_max, colors="#1f77b4", linewidths=5, alpha=0.75)
        drawn.append((by, r_min, r_max))

    # ── seta de CD (canto inferior esquerdo) ─────────────────────────────
    cd_y  = -0.02
    cd_x0 = ax_x0
    cd_x1 = ax_x0 + cd
    ax.annotate("", xy=(cd_x1, cd_y), xytext=(cd_x0, cd_y),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.8))
    ax.text((cd_x0 + cd_x1) / 2, cd_y + 0.06,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8.5, color="crimson")

    ax.set_title(
        f"Critical Difference Diagram — {label}\n"
        f"k={k} modelos · α=0.05",
        fontsize=10, pad=6,
    )

    plt.tight_layout()
    out_path = PLOTS_DIR / f"cd_diagram_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CD diagram salvo em: {out_path}")


# ── Boxplot de rankings ───────────────────────────────────────────────────────

def boxplot_rankings(nrmse_matrix: pd.DataFrame, label: str, slug: str):
    """
    Para cada série, ranqueia os modelos por nRMSE (rank 1 = melhor).
    Gera boxplot dos rankings e salva em analysis/plots/.
    """
    clean = nrmse_matrix.dropna()
    # rank 1 = menor nRMSE (melhor)
    ranks = clean.rank(axis=1, method="average", ascending=True)

    n_models = len(ranks.columns)
    order    = ranks.median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), 5))

    data_plot = [ranks[col].values for col in order]
    bp = ax.boxplot(
        data_plot,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        notch=False,
    )

    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n_models))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel("Ranking (1 = melhor)", fontsize=10)
    ax.set_title(f"Distribuição de Rankings — {label}", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


    plt.tight_layout()
    out_path = PLOTS_DIR / f"boxplot_rankings_{slug}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Boxplot salvo em: {out_path}")

    # tabela de resumo de rankings
    print(f"\n  Resumo de rankings (mediana | média):")
    summary = pd.DataFrame({
        "mediana_rank": ranks.median().round(2),
        "media_rank"  : ranks.mean().round(2),
        "rank_1_pct"  : (ranks == 1).mean().mul(100).round(1),
    }).sort_values("mediana_rank")
    print(summary.to_string())


# ── Boxplot organizado por paradigma ─────────────────────────────────────────

# Ordem e cores por paradigma
_PARADIGMA_CONFIG = {
    "Individual": {
        "modelos" : ["ARIMA", "MLP", "LR", "SVR"],
        "cor"     : "#2ca02c",   # verde
        "label"   : "Individual",
    },
    "Local": {
        "modelos" : ["Local_Ridge", "Local_SVR", "Local_MLP", "Local_LGBM"],
        "cor"     : "#1f77b4",   # azul
        "label"   : "Local",
    },
    "Global": {
        "modelos" : ["Glob_Ridge", "Glob_RF", "Glob_MLP", "Glob_LGBM", "Glob_Chronos"],
        "cor"     : "#d62728",   # vermelho
        "label"   : "Global",
    },
}


def boxplot_rankings_por_paradigma(nrmse_matrix: pd.DataFrame, label: str, slug: str):
    """
    Boxplot de rankings organizado por paradigma (Individual | Local | Global).
    Modelos ausentes na matriz são ignorados silenciosamente.
    """
    clean = nrmse_matrix.dropna()
    ranks = clean.rank(axis=1, method="average", ascending=True)

    # monta ordem e metadados apenas com modelos presentes
    ordem, cores, xticks_paradigma = [], [], {}
    pos_start = 0
    for paradigma, cfg in _PARADIGMA_CONFIG.items():
        presentes = [m for m in cfg["modelos"] if m in ranks.columns]
        if not presentes:
            continue
        xticks_paradigma[paradigma] = {
            "cor"   : cfg["cor"],
            "inicio": pos_start,
            "fim"   : pos_start + len(presentes) - 1,
        }
        ordem.extend(presentes)
        cores.extend([cfg["cor"]] * len(presentes))
        pos_start += len(presentes)

    if not ordem:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(ordem) * 1.1), 5))

    data_plot = [ranks[m].values for m in ordem]
    bp = ax.boxplot(
        data_plot,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        widths=0.6,
    )
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)

    # remove prefixos "Local_" / "Glob_" nos rótulos do eixo X
    def _strip_prefix(name: str) -> str:
        for pfx in ("Local_", "Glob_"):
            if name.startswith(pfx):
                return name[len(pfx):]
        return name

    ax.set_xticks(range(1, len(ordem) + 1))
    ax.set_xticklabels([_strip_prefix(m) for m in ordem], rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel("Ranking (1 = melhor)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # separadores verticais entre paradigmas
    for paradigma, info in xticks_paradigma.items():
        ini = info["inicio"]
        if ini > 0:
            ax.axvline(ini + 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    # rótulos de paradigma em coordenadas de figura, abaixo dos tick labels
    fig_width = fig.get_size_inches()[0]
    n_total   = len(ordem)
    for paradigma, info in xticks_paradigma.items():
        ini, fim = info["inicio"], info["fim"]
        # converte posição central para fração da figura
        cx_data  = (ini + fim) / 2 + 1          # posição em unidades de boxplot
        cx_frac  = ax.transData.transform((cx_data, 0))[0] / (fig.dpi * fig_width)
        fig.text(cx_frac, 0.04, paradigma,
                 ha="center", va="bottom", fontsize=10,
                 color=info["cor"], fontweight="bold")
    out_path = PLOTS_DIR / f"boxplot_paradigmas_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplot por paradigma salvo em: {out_path}")


# ── Construtores de matrizes (com nRMSE) ─────────────────────────────────────

def matriz_individuais(metricas_path: Path, normas: dict) -> pd.DataFrame:
    df = pd.read_csv(metricas_path)
    df = df[df["Modelo"].isin(["ARIMA", "LR", "SVR", "MLP"])].copy()
    pv = df.pivot_table(index="Codigo", columns="Modelo", values="RMSE", aggfunc="first")
    pv = pv[["ARIMA", "LR", "SVR", "MLP"]].reset_index()
    pv = normalizar_df(pv, ["ARIMA", "LR", "SVR", "MLP"], normas)
    return pv.set_index("Codigo")


def matriz_paradigmas(unificada_path: Path, normas: dict) -> pd.DataFrame:
    df  = pd.read_csv(unificada_path)
    out = df[["Codigo"]].copy()
    out["Ind"]     = df["Ind_RMSE"].values
    out["Local"]   = df["Pooled_Melhor_RMSE"].values
    out["Global"]  = df["Global_Melhor_RMSE"].values
    if "Global_Chronos_RMSE" in df.columns:
        out["Chronos"] = df["Global_Chronos_RMSE"].values
    rmse_cols = [c for c in ["Ind", "Local", "Global", "Chronos"] if c in out.columns]
    out = normalizar_df(out, rmse_cols, normas)
    return out.set_index("Codigo")


def matriz_todos(metricas_path: Path, unificada_path: Path, normas: dict,
                 tem_lgbm_cluster: bool = False) -> pd.DataFrame:
    ind = matriz_individuais(metricas_path, normas)

    uni = pd.read_csv(unificada_path).set_index("Codigo")
    local_cols = {
        "Local_Ridge": "Pooled_Ridge_RMSE",
        "Local_SVR"  : "Pooled_SVR_RMSE",
        "Local_MLP"  : "Pooled_MLP_RMSE",
    }
    global_cols = {
        "Glob_Ridge"   : "Global_Ridge_RMSE",
        "Glob_RF"      : "Global_RF_RMSE",
        "Glob_MLP"     : "Global_MLP_RMSE",
        "Glob_LGBM"    : "Global_LGBM_RMSE",
        "Glob_Chronos" : "Global_Chronos_RMSE",
    }
    extra = {}
    for label, col in {**local_cols, **global_cols}.items():
        if col in uni.columns:
            extra[label] = uni[col]
    if tem_lgbm_cluster and "LGBM_Cluster_RMSE" in uni.columns:
        extra["Local_LGBM"] = uni["LGBM_Cluster_RMSE"]

    extra_df = pd.DataFrame(extra)
    # normalizar colunas extra
    extra_df = extra_df.reset_index()
    extra_df = normalizar_df(extra_df, list(extra.keys()), normas)
    extra_df = extra_df.set_index("Codigo")

    return ind.join(extra_df, how="inner")


# ── Análise principal ─────────────────────────────────────────────────────────

def analisar(cfg: dict, tem_lgbm_cluster: bool = False):
    label = cfg["label"]
    slug  = cfg["slug"]
    separador(f"EXPERIMENTO: {label}", char="═")

    print(f"\n  Calculando fatores de normalização (mean|y_treino|) …")
    normas = calcular_normas(cfg["parquet"], cfg["col_valor"], cfg["unificada"])
    print(f"  {len(normas)} séries com fator de normalização calculado.")

    met_path = cfg["metricas"]
    uni_path = cfg["unificada"]
    excluir  = cfg.get("excluir", set())

    def _filtra(m):
        if excluir:
            return m.drop(index=[c for c in excluir if c in m.index], errors="ignore")
        return m

    # ── Bloco A: Modelos Individuais ─────────────────────────────────────────
    sub_separador("A. MODELOS INDIVIDUAIS  (ARIMA · LR · SVR · MLP)")
    mat_a = _filtra(matriz_individuais(met_path, normas))
    stat_a, p_a = friedman_test(mat_a, "modelos individuais")
    if not np.isnan(p_a) and p_a < ALPHA:
        nemenyi_posthoc(mat_a, "individuais")
    wilcoxon_pairwise(mat_a, "individuais")
    cd_diagram(mat_a, f"{label} — Modelos Individuais", f"{slug}_individuais")
    boxplot_rankings(mat_a, f"{label} — Individuais", f"{slug}_individuais")

    # ── Bloco B: Paradigmas ──────────────────────────────────────────────────
    sub_separador("B. PARADIGMAS  (Melhor Individual · Melhor Local · Melhor Global · Chronos)")
    mat_b = _filtra(matriz_paradigmas(uni_path, normas))
    stat_b, p_b = friedman_test(mat_b, "paradigmas")
    if not np.isnan(p_b) and p_b < ALPHA:
        nemenyi_posthoc(mat_b, "paradigmas")
    wilcoxon_pairwise(mat_b, "paradigmas")
    cd_diagram(mat_b, f"{label} — Paradigmas", f"{slug}_paradigmas")
    boxplot_rankings(mat_b, f"{label} — Paradigmas", f"{slug}_paradigmas")

    # ── Bloco C: Todos os modelos ────────────────────────────────────────────
    sub_separador("C. TODOS OS MODELOS  (comparação global)")
    mat_c = _filtra(matriz_todos(met_path, uni_path, normas, tem_lgbm_cluster))
    stat_c, p_c = friedman_test(mat_c, "todos os modelos")
    if not np.isnan(p_c) and p_c < ALPHA:
        nemenyi_posthoc(mat_c, "todos os modelos", indent=2)
    wilcoxon_pairwise(mat_c, "todos os modelos", indent=2)
    cd_diagram(mat_c, f"{label} — Todos os Modelos", f"{slug}_todos")
    boxplot_rankings(mat_c, f"{label} — Todos os Modelos", f"{slug}_todos")
    boxplot_rankings_por_paradigma(mat_c, label, f"{slug}_todos")

    return mat_a, mat_b, mat_c

    print()


def nota_n_pequeno(n: int):
    print(f"\n  ⚠  ATENÇÃO: n={n} é pequeno. Wilcoxon requer n≥6 por par; Nemenyi perde poder.")
    print("     Resultados são indicativos; poder estatístico é baixo.")


# ── Experimento combinado ─────────────────────────────────────────────────────

def analisar_combinado(mat_rec: pd.DataFrame, mat_desp: pd.DataFrame,
                       bloco: str, slug_bloco: str):
    """
    Concatena receita e despesa e roda Friedman + Nemenyi + CD diagram
    para o conjunto unificado (apenas colunas comuns).
    """
    cols_comuns = mat_rec.columns.intersection(mat_desp.columns).tolist()
    if not cols_comuns:
        print(f"  [combinado/{bloco}] sem colunas comuns — pulado.")
        return

    combined = pd.concat(
        [mat_rec[cols_comuns], mat_desp[cols_comuns]],
        ignore_index=True,
    ).dropna()

    label = f"Combinado Receita+Despesa (n={len(combined)}) — {bloco}"
    slug  = f"combinado_{slug_bloco}"

    sub_separador(f"COMBINADO — {bloco}  (n={len(combined)}, k={len(cols_comuns)})")
    stat, p = friedman_test(combined, bloco)
    if not np.isnan(p) and p < ALPHA:
        nemenyi_posthoc(combined, bloco)
    wilcoxon_pairwise(combined, bloco)
    cd_diagram(combined, label, slug)
    boxplot_rankings(combined, label, slug)

    # Para o boxplot por paradigma, inclui colunas extras de receita (ex: LGBM_Clust)
    # que não existem em despesa — as linhas de despesa ficam NaN e são descartadas
    # pelo dropna() interno do ranking.
    cols_extras = [c for c in mat_rec.columns if c not in cols_comuns]
    if cols_extras:
        combined_ext = pd.concat(
            [mat_rec[cols_comuns + cols_extras],
             mat_desp[cols_comuns].reindex(columns=cols_comuns + cols_extras)],
            ignore_index=True,
        )
    else:
        combined_ext = combined
    boxplot_rankings_por_paradigma(combined_ext, label, slug)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("  COMPARAÇÃO ESTATÍSTICA DE MODELOS DE PREVISÃO")
    print("  Protocolo: Friedman + Nemenyi + Wilcoxon/Holm-Bonferroni")
    print("  Métrica: nRMSE = RMSE / mean(|y_treino|)")
    print("  Referência: Demšar (2006), J. Machine Learning Research 7:1–30")
    print("█" * 80)

    mat_rec_a, mat_rec_b, mat_rec_c = analisar(RECEITA, tem_lgbm_cluster=True)

    separador("NOTA METODOLÓGICA — DESPESAS", char="─")
    nota_n_pequeno(19)
    mat_des_a, mat_des_b, mat_des_c = analisar(DESPESA, tem_lgbm_cluster=False)

    separador("EXPERIMENTO COMBINADO  (Receita + Despesa)", char="═")
    analisar_combinado(mat_rec_a, mat_des_a, "Modelos Individuais", "individuais")
    analisar_combinado(mat_rec_b, mat_des_b, "Paradigmas",          "paradigmas")
    analisar_combinado(mat_rec_c, mat_des_c, "Todos os Modelos",    "todos")
