"""
Gráficos reutilizáveis do experimento de receitas. Todos são salvos em PNG na
pasta results/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORES, COR_MODELO, PLOTS_DIR

C = CORES


# Utilitários internos

def _formatar_eixo_y(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"R${x/1e6:.1f}Mi" if abs(x) >= 1e6 else f"R${x:,.0f}"
        )
    )


def _salvar(fig, nome: str, mostrar: bool = False) -> Path:
    """Salva a figura em PLOTS_DIR como PNG e fecha."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = PLOTS_DIR / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    if mostrar:
        plt.show()
    plt.close(fig)
    return caminho


def _badge_sinal(grupo: str) -> str:
    return {
        "apenas_positivos"      : "✔ Apenas Positivos",
        "apenas_negativos"      : "✖ Apenas Negativos (shift aplicado)",
        "positivos_e_negativos" : "± Positivos e Negativos",
    }.get(grupo, grupo)


def _estilo_base():
    plt.rcParams.update({
        "font.family"      : "DejaVu Sans",
        "axes.facecolor"   : C["cinza_cla"],
        "figure.facecolor" : C["branco"],
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.edgecolor"   : C["cinza_med"],
        "axes.grid"        : True,
        "grid.color"       : C["branco"],
        "grid.linewidth"   : 0.8,
    })


# Gráfico por série

def plotar_serie(
    codigo: str,
    nome: str,
    grupo_sinal: str,
    treino_idx,
    treino_vals: np.ndarray,
    teste_idx,
    teste_vals: np.ndarray,
    resultados_serie: dict,
    melhor_modelo: str,
    meses_prev: list,
    fc_vals: list,
    n_holdout: int,
    n_lags: int,
    mostrar: bool = False,
    sufixo: str = "",
) -> Path:
    """Painel duplo (histórico + previsão no conjunto de teste + forecast/resíduos), salvo
    em results/plots/serie_{codigo}{sufixo}.png. O `sufixo` (ex.: "_local",
    "_global") evita sobrescrever os gráficos de outros experimentos."""
    _estilo_base()
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), facecolor=C["branco"],
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax = axes[0]

    # Histórico real
    ax.plot(treino_idx.astype(str), treino_vals,
            color=C["azul_med"], linewidth=1.8, label="Treino (real)", zorder=3)
    ax.plot(teste_idx.astype(str), teste_vals,
            color=C["azul_esc"], linewidth=2.0, linestyle="--",
            label="Teste (real)", zorder=4)

    # Previsões de todos os modelos no conjunto de teste
    for mod, res in resultados_serie.items():
        destaque = mod == melhor_modelo
        ax.plot(
            teste_idx.astype(str), res["preds_teste"],
            color=COR_MODELO.get(mod, "#555"),
            linewidth=2.0 if destaque else 1.2,
            alpha=1.0 if destaque else 0.55,
            linestyle="-" if destaque else ":",
            marker="o", markersize=4 if destaque else 2,
            label=(
                f"{'★ ' if destaque else ''}{mod} — "
                f"RMSE={res['RMSE']:,.0f}  MAPE={res['MAPE']:.1f}%"
            ),
            zorder=5 if destaque else 3,
        )

    # Forecast final
    ax.plot(
        [str(m) for m in meses_prev], fc_vals,
        color=C["verde"], linewidth=2.4, marker="*", markersize=8, zorder=6,
        label=f"Forecast {melhor_modelo} ({len(meses_prev)}m)",
    )
    ax.axvline(x=str(teste_idx[0]), color="gray",
               linewidth=1.0, linestyle="--", alpha=0.5, label="Início conjunto de teste")

    ax.set_title(
        f"Fonte {codigo} — {_badge_sinal(grupo_sinal)}\n{nome[:90]}",
        fontsize=10, fontweight="bold", color=C["azul_esc"],
    )
    ax.set_ylabel("Receita Realizada (R$)")
    _formatar_eixo_y(ax)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.legend(fontsize=8, loc="upper left")

    # Resíduos
    ax2      = axes[1]
    residuos = teste_vals - np.array(resultados_serie[melhor_modelo]["preds_teste"])
    ax2.bar(
        teste_idx.astype(str), residuos,
        color=[C["verde"] if r >= 0 else C["vermelho"] for r in residuos],
        alpha=0.8,
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Resíduos — {melhor_modelo} (selecionado) | conjunto de teste={n_holdout}  lags={n_lags}",
        fontsize=9, fontweight="bold", color=C["azul_esc"],
    )
    ax2.set_ylabel("Erro (R$)")
    ax2.tick_params(axis="x", rotation=45, labelsize=6)
    _formatar_eixo_y(ax2)

    plt.tight_layout()
    return _salvar(fig, f"serie_{codigo}{sufixo}", mostrar=mostrar)


# Resumo global dos modelos

def plotar_resumo_modelos(metricas_df: pd.DataFrame, mostrar: bool = False) -> Path:
    """Barras: RMSE e MAE médios por modelo."""
    _estilo_base()
    resumo = (
        metricas_df.groupby("Modelo")[["RMSE", "MAE"]]
        .mean().reset_index().sort_values("RMSE")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C["branco"])

    for ax, metrica in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(
            resumo["Modelo"], resumo[metrica],
            color=[COR_MODELO.get(m, C["azul_med"]) for m in resumo["Modelo"]],
            edgecolor=C["branco"], width=0.5,
        )
        for bar, val in zip(bars, resumo[metrica]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:,.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_title(f"{metrica} Médio por Modelo", fontsize=11,
                     fontweight="bold", color=C["azul_esc"])
        ax.set_ylabel(metrica)
        ax.set_facecolor(C["cinza_cla"])
        _formatar_eixo_y(ax)

    plt.tight_layout()
    return _salvar(fig, "resumo_modelos", mostrar=mostrar)


# Distribuição de vencedores

def plotar_vencedores(prev_df: pd.DataFrame, mostrar: bool = False) -> Path:
    """Contagem e distribuição por grupo de sinal dos modelos vencedores."""
    _estilo_base()
    vencedores = prev_df["Melhor_Modelo"].value_counts()
    fig, axes  = plt.subplots(1, 2, figsize=(12, 4), facecolor=C["branco"])

    ax = axes[0]
    bars = ax.bar(
        vencedores.index, vencedores.values,
        color=[COR_MODELO.get(m, C["azul_med"]) for m in vencedores.index],
        edgecolor=C["branco"], width=0.5,
    )
    for bar, val in zip(bars, vencedores.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, str(val),
                ha="center", fontweight="bold")
    ax.set_title("Modelos Vencedores — Contagem", fontweight="bold", color=C["azul_esc"])
    ax.set_facecolor(C["cinza_cla"])

    ax2 = axes[1]
    por_sinal = (
        prev_df.groupby(["Grupo_Sinal", "Melhor_Modelo"])
        .size().unstack(fill_value=0)
    )
    por_sinal.plot(
        kind="bar", ax=ax2,
        color=[COR_MODELO.get(c, C["azul_med"]) for c in por_sinal.columns],
        edgecolor=C["branco"],
    )
    ax2.set_title("Vencedores por Grupo de Sinal", fontweight="bold", color=C["azul_esc"])
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend(title="Modelo")
    ax2.set_facecolor(C["cinza_cla"])

    plt.tight_layout()
    return _salvar(fig, "vencedores", mostrar=mostrar)


# Boxplot de RMSE

def plotar_distribuicao_rmse(metricas_df: pd.DataFrame, mostrar: bool = False) -> Path:
    """Boxplot da distribuição de RMSE por modelo em todas as séries."""
    _estilo_base()
    modelos = sorted(metricas_df["Modelo"].unique())
    dados   = [metricas_df[metricas_df["Modelo"] == m]["RMSE"].values for m in modelos]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C["branco"])
    bp = ax.boxplot(dados, labels=modelos, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, mod in zip(bp["boxes"], modelos):
        patch.set_facecolor(COR_MODELO.get(mod, C["azul_med"]))
        patch.set_alpha(0.75)

    ax.set_title("Distribuição de RMSE por Modelo (todas as séries)",
                 fontsize=12, fontweight="bold", color=C["azul_esc"])
    ax.set_ylabel("RMSE")
    ax.set_facecolor(C["cinza_cla"])
    _formatar_eixo_y(ax)
    plt.tight_layout()
    return _salvar(fig, "distribuicao_rmse", mostrar=mostrar)


# Ablation: número de lags

def plotar_ablation_lags(lags_df: pd.DataFrame, mostrar: bool = False) -> Path:
    """
    Linha: RMSE médio de cada modelo × número de lags.
    ARIMA é incluído como referência horizontal (não usa lags).
    """
    _estilo_base()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["branco"])

    ax = axes[0]
    for mod, grupo in lags_df.groupby("Modelo"):
        resumo = grupo.groupby("N_Lags")["RMSE"].mean().reset_index()
        if mod == "ARIMA":
            rmse_arima = lags_df[lags_df["Modelo"] == "ARIMA"]["RMSE"].mean()
            lags_unicos = sorted(lags_df["N_Lags"].unique())
            ax.hlines(rmse_arima, lags_unicos[0], lags_unicos[-1],
                      colors=COR_MODELO.get(mod, "#555"),
                      linewidth=1.8, linestyle="--",
                      label=f"ARIMA (ref.) — RMSE={rmse_arima:,.0f}")
        else:
            ax.plot(resumo["N_Lags"], resumo["RMSE"],
                    marker="o", linewidth=2,
                    color=COR_MODELO.get(mod, C["azul_med"]), label=mod)

    ax.set_title("Ablation — RMSE Médio × Número de Lags",
                 fontsize=11, fontweight="bold", color=C["azul_esc"])
    ax.set_xlabel("Número de Lags")
    ax.set_ylabel("RMSE Médio")
    ax.legend(title="Modelo", fontsize=9)
    ax.set_facecolor(C["cinza_cla"])
    _formatar_eixo_y(ax)

    ax2 = axes[1]
    for mod, grupo in lags_df[lags_df["Modelo"] != "ARIMA"].groupby("Modelo"):
        resumo = grupo.groupby("N_Lags")["RMSE"].mean().reset_index()
        base   = resumo["RMSE"].min()
        resumo["Var_Pct"] = (resumo["RMSE"] - base) / base * 100
        ax2.plot(resumo["N_Lags"], resumo["Var_Pct"],
                 marker="o", linewidth=2,
                 color=COR_MODELO.get(mod, C["azul_med"]), label=mod)

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_title("Variação % vs Melhor Configuração de Lags",
                  fontsize=11, fontweight="bold", color=C["azul_esc"])
    ax2.set_xlabel("Número de Lags")
    ax2.set_ylabel("Variação % do RMSE")
    ax2.legend(title="Modelo", fontsize=9)
    ax2.set_facecolor(C["cinza_cla"])

    plt.tight_layout()
    return _salvar(fig, "ablation_lags", mostrar=mostrar)


# Ablation: tamanho do conjunto de teste

def plotar_ablation_holdout(holdout_df: pd.DataFrame, mostrar: bool = False) -> Path:
    """
    Linha: RMSE médio de cada modelo × tamanho do conjunto de teste.
    Painel de ranking por conjunto de teste para identificar inversões de posição.
    """
    _estilo_base()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C["branco"])

    ax = axes[0]
    for mod, grupo in holdout_df.groupby("Modelo"):
        resumo = grupo.groupby("N_Holdout")["RMSE"].mean().reset_index()
        ax.plot(resumo["N_Holdout"], resumo["RMSE"],
                marker="s", linewidth=2,
                color=COR_MODELO.get(mod, C["azul_med"]), label=mod)

    ax.set_title("Ablation — RMSE Médio × Tamanho do Conjunto de teste",
                 fontsize=11, fontweight="bold", color=C["azul_esc"])
    ax.set_xlabel("Tamanho do Conjunto de teste (meses)")
    ax.set_ylabel("RMSE Médio")
    ax.legend(title="Modelo", fontsize=9)
    ax.set_facecolor(C["cinza_cla"])
    _formatar_eixo_y(ax)

    ax2       = axes[1]
    holdouts  = sorted(holdout_df["N_Holdout"].unique())
    modelos   = sorted(holdout_df["Modelo"].unique())
    rankings  = np.zeros((len(modelos), len(holdouts)))

    for j, h in enumerate(holdouts):
        sub     = holdout_df[holdout_df["N_Holdout"] == h]
        rmses   = sub.groupby("Modelo")["RMSE"].mean()
        ordered = rmses.rank().reindex(modelos, fill_value=len(modelos))
        for i, mod in enumerate(modelos):
            rankings[i, j] = ordered.get(mod, len(modelos))

    im = ax2.imshow(rankings, cmap="RdYlGn_r", aspect="auto",
                    vmin=1, vmax=len(modelos))
    ax2.set_xticks(range(len(holdouts)))
    ax2.set_xticklabels([f"{h}m" for h in holdouts])
    ax2.set_yticks(range(len(modelos)))
    ax2.set_yticklabels(modelos)
    ax2.set_title("Ranking dos Modelos por Conjunto de teste\n(verde = 1º lugar)",
                  fontsize=10, fontweight="bold", color=C["azul_esc"])
    ax2.set_xlabel("Tamanho do Conjunto de teste")

    for i in range(len(modelos)):
        for j in range(len(holdouts)):
            ax2.text(j, i, f"{int(rankings[i,j])}º",
                     ha="center", va="center", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax2, label="Posição no ranking")
    plt.tight_layout()
    return _salvar(fig, "ablation_holdout", mostrar=mostrar)
