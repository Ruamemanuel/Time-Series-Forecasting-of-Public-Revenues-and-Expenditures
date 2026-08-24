"""
Quatro formas de clusterizar as séries, todas usando só a janela de treino:

  features    K-Means sobre 8 features estatísticas (escala, variabilidade,
              tendência, ACF, sazonalidade, assimetria).
  correlacao  correlação de Pearson par a par + hierárquico (UPGMA); agrupa
              séries que sobem e descem juntas.
  cosseno     similaridade de cosseno numa janela recente fixa + UPGMA; dá mais
              peso ao comportamento atual (equivale à correlação nessa janela).
  dtw         Dynamic Time Warping par a par, com as séries em z-score, + UPGMA;
              compara a forma e tolera pequenos deslocamentos no tempo. Banda de
              Sakoe-Chiba de 10% do par (mín. 3 pontos), distância normalizada
              para [0,1].
  auto        roda os quatro e fica com o de maior silhouette.

O k é escolhido em [k_min, k_max] pelo silhouette, com a métrica certa de cada
método (euclidiano nas features; precomputed nos demais). O embedding_2d para
visualização vem de PCA (features) ou MDS clássico/PCoA (matrizes de distância).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Dataclass de resultado

@dataclass
class ResultadoClustering:
    """Resultado completo do processo de clustering."""
    labels           : np.ndarray         # label por série (mesma ordem de codigos)
    codigos          : list               # códigos das séries
    k                : int                # número de clusters escolhido
    silhouette       : float              # silhouette score do melhor k
    features_scaled  : np.ndarray         # features z-score ou matriz de distâncias NxN
    cluster_map      : dict               # {codigo: cluster_id}
    cluster_members  : dict               # {cluster_id: [codigos]}
    silhouette_per_k : dict = field(default_factory=dict)   # {k: score}
    embedding_2d     : Optional[np.ndarray] = None          # 2D para visualização
    metodo           : str = "features"   # método utilizado
    dist_matrix      : Optional[np.ndarray] = None          # matriz de distâncias (se aplicável)


# Visualização 2D (MDS clássico / PCoA)

def _classical_mds(dist_mat: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Multi-Dimensional Scaling clássico (PCoA) a partir de uma matriz de distâncias.
    O(n²) em memória, rápido para n ≤ 1000.
    """
    n   = dist_mat.shape[0]
    D2  = dist_mat ** 2
    H   = np.eye(n) - np.ones((n, n)) / n           # centering matrix
    B   = -0.5 * H @ D2 @ H
    eigvals, eigvecs = np.linalg.eigh(B)

    # sort descending
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    coords = np.zeros((n, n_components))
    for i in range(n_components):
        if i < n and eigvals[i] > 0:
            coords[:, i] = eigvecs[:, i] * np.sqrt(eigvals[i])
    return coords


# Clustering hierárquico (UPGMA) sobre matriz de distâncias precomputada

def _hierarchical_cluster(
    dist_mat    : np.ndarray,
    codigos     : list,
    k_min       : int,
    k_max       : int,
) -> Tuple[np.ndarray, int, float, dict]:
    """Hierárquico (UPGMA) sobre a matriz de distâncias, escolhendo k pelo
    silhouette. Devolve (best_labels, best_k, best_score, scores_por_k)."""
    n          = len(codigos)
    k_max_real = min(k_max, n - 1)

    # condensed form para linkage
    condensed  = squareform(dist_mat, checks=False)
    Z          = linkage(condensed, method="average")   # UPGMA

    best_k      = k_min
    best_score  = -1.0
    best_labels = None
    scores_k    : dict = {}

    for k in range(k_min, k_max_real + 1):
        lbls = fcluster(Z, t=k, criterion="maxclust") - 1   # 0-indexed
        if len(np.unique(lbls)) < 2:
            continue
        score       = float(silhouette_score(dist_mat, lbls, metric="precomputed"))
        scores_k[k] = score
        logger.debug("hierarquico k=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = lbls.copy()

    if best_labels is None:
        best_labels = fcluster(Z, t=k_min, criterion="maxclust") - 1
        best_k      = k_min
        best_score  = 0.0

    return best_labels, best_k, best_score, scores_k


def _make_resultado(
    labels       : np.ndarray,
    codigos      : list,
    k            : int,
    silhouette   : float,
    scores_k     : dict,
    features_mat : np.ndarray,
    embedding    : np.ndarray,
    metodo       : str,
    dist_matrix  : Optional[np.ndarray] = None,
) -> ResultadoClustering:
    """Helper para montar ResultadoClustering com cluster_map / cluster_members."""
    cluster_map     : dict = {cod: int(lbl) for cod, lbl in zip(codigos, labels)}
    cluster_members : dict = {}
    for cod, lbl in cluster_map.items():
        cluster_members.setdefault(lbl, []).append(cod)

    sizes = {cid: len(m) for cid, m in cluster_members.items()}
    logger.info(
        "Clustering [%s]: k=%d (silhouette=%.4f) | series por cluster: %s",
        metodo, k, silhouette, sizes,
    )

    return ResultadoClustering(
        labels           = labels,
        codigos          = codigos,
        k                = k,
        silhouette       = silhouette,
        features_scaled  = features_mat,
        cluster_map      = cluster_map,
        cluster_members  = cluster_members,
        silhouette_per_k = scores_k,
        embedding_2d     = embedding,
        metodo           = metodo,
        dist_matrix      = dist_matrix,
    )


# Método 1 · K-Means sobre features estatísticas

def clusterizar_series(
    features_df  : pd.DataFrame,
    k_min        : int = 2,
    k_max        : int = 6,
    random_state : int = 42,
    n_init       : int = 20,
) -> ResultadoClustering:
    """
    Clustering K-Means sobre 8 features estatísticas extraídas do treino.

    Features: log_mean_abs, cv, trend_slope_norm, acf_1/2/3, seasonal_strength, skewness.
    Silhouette calculado com distância Euclidiana sobre features z-score.
    embedding_2d: PCA(2) sobre features z-score.
    """
    n = len(features_df)
    k_max_real = min(k_max, n - 1)

    if k_max_real < k_min:
        labels = np.zeros(n, dtype=int)
        codigos = list(features_df.index)
        X = np.zeros((n, features_df.shape[1]))
        emb = np.zeros((n, 2))
        cluster_map     = {cod: 0 for cod in codigos}
        cluster_members = {0: codigos[:]}
        return ResultadoClustering(
            labels=labels, codigos=codigos, k=1, silhouette=0.0,
            features_scaled=X, cluster_map=cluster_map,
            cluster_members=cluster_members,
            embedding_2d=emb, metodo="features",
        )

    codigos = list(features_df.index)
    scaler  = StandardScaler()
    X       = scaler.fit_transform(features_df.fillna(0.0).values)

    best_k      = k_min
    best_score  = -1.0
    best_labels = None
    scores_k    : dict = {}

    for k in range(k_min, k_max_real + 1):
        km     = KMeans(n_clusters=k, random_state=random_state,
                        n_init=n_init, max_iter=500)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        score       = float(silhouette_score(X, labels))
        scores_k[k] = score
        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = labels.copy()

    if best_labels is None:
        km          = KMeans(n_clusters=k_min, random_state=random_state, n_init=n_init)
        best_labels = km.fit_predict(X)
        best_k      = k_min
        best_score  = 0.0

    # PCA 2D para visualização
    try:
        from sklearn.decomposition import PCA as _PCA
        emb = _PCA(n_components=2, random_state=random_state).fit_transform(X)
    except Exception:
        emb = X[:, :2] if X.shape[1] >= 2 else np.zeros((n, 2))

    return _make_resultado(
        labels=best_labels, codigos=codigos, k=best_k,
        silhouette=best_score, scores_k=scores_k,
        features_mat=X, embedding=emb,
        metodo="features",
    )


# Método 2 · Correlação de Pearson + hierárquico

def _build_corr_matrix(
    treinos  : dict,
    codigos  : list,
    min_overlap: int = 6,
) -> np.ndarray:
    """
    Matriz de correlação de Pearson N×N.
    Para cada par (i, j): correlação nos últimos min(len_i, len_j) períodos.
    Séries com sobreposição < min_overlap recebem corr = 0.
    """
    n    = len(codigos)
    corr = np.zeros((n, n))
    np.fill_diagonal(corr, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            vi = np.asarray(treinos[codigos[i]], dtype=float)
            vj = np.asarray(treinos[codigos[j]], dtype=float)
            overlap = min(len(vi), len(vj))
            if overlap < min_overlap:
                c = 0.0
            else:
                vi_ = vi[-overlap:]
                vj_ = vj[-overlap:]
                c = float(np.corrcoef(vi_, vj_)[0, 1])
                if np.isnan(c) or np.isinf(c):
                    c = 0.0
            corr[i, j] = c
            corr[j, i] = c

    return corr


def clusterizar_por_correlacao(
    treinos      : dict,
    k_min        : int = 2,
    k_max        : int = 6,
    min_overlap  : int = 6,
    random_state : int = 42,
) -> ResultadoClustering:
    """
    Clustering baseado em correlação de Pearson par a par.

    Distância: d(i,j) = (1 − corr(i,j)) / 2  ∈ [0, 1]
      - corr=+1 (séries idênticas)         : d = 0
      - corr= 0 (não correlacionadas)      : d = 0.5
      - corr=-1 (anti-correlacionadas)     : d = 1
    Clustering: UPGMA (average linkage) sobre a matriz de distâncias.
    Silhouette: precomputed.
    embedding_2d: MDS clássico (PCoA).
    """
    codigos  = sorted(str(c) for c in treinos.keys())
    treinos_ = {str(c): treinos[c] for c in treinos}

    corr_mat = _build_corr_matrix(treinos_, codigos, min_overlap=min_overlap)
    dist_mat = (1.0 - corr_mat) / 2.0
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = np.clip(dist_mat, 0.0, 1.0)

    best_labels, best_k, best_score, scores_k = _hierarchical_cluster(
        dist_mat, codigos, k_min, k_max
    )

    emb = _classical_mds(dist_mat, n_components=2)

    return _make_resultado(
        labels=best_labels, codigos=codigos, k=best_k,
        silhouette=best_score, scores_k=scores_k,
        features_mat=dist_mat, embedding=emb,
        metodo="correlacao", dist_matrix=dist_mat,
    )


# Método 3 · Cosseno (janela recente fixa) + hierárquico

def _build_cosine_matrix(
    treinos  : dict,
    codigos  : list,
    n_common : int = 36,
) -> np.ndarray:
    """
    Matriz de similaridade de cosseno N×N.
    Cada série é representada como vetor z-score dos últimos n_common períodos
    (preenchido com zeros à esquerda para séries menores que n_common).

    Nota: cosine similarity de vetores z-score = correlação de Pearson na janela fixa.
    A diferença em relação ao método de correlação:
      - usa janela FIXA (n_common) em vez de janela variável
      - enfatiza comportamento recente, mais robusto a quebras estruturais antigas
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n   = len(codigos)
    mat = np.zeros((n, n_common))

    for i, cod in enumerate(codigos):
        v   = np.asarray(treinos[cod], dtype=float)
        mu  = float(np.mean(v))
        std = float(np.std(v, ddof=1)) if len(v) > 1 else 1.0
        if std < 1e-9:
            std = 1.0
        v_z = (v - mu) / std
        l   = min(len(v_z), n_common)
        mat[i, n_common - l:] = v_z[-l:]   # alinha pelo lado direito (mais recente)

    sim_mat  = cosine_similarity(mat)                    # [-1, 1]
    sim_mat  = np.clip(sim_mat, -1.0, 1.0)
    dist_mat = 1.0 - sim_mat                             # [0, 2]
    dist_mat = np.clip(dist_mat / 2.0, 0.0, 1.0)        # normaliza para [0, 1]
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def clusterizar_por_cosseno(
    treinos      : dict,
    k_min        : int = 2,
    k_max        : int = 6,
    n_common     : int = 36,
    random_state : int = 42,
) -> ResultadoClustering:
    """
    Clustering baseado em similaridade de cosseno sobre janela recente fixa.

    Distância: d(i,j) = (1 − cosine_sim(i,j)) / 2  ∈ [0, 1]
    Clustering: UPGMA (average linkage) sobre a matriz de distâncias.
    Silhouette: precomputed.
    embedding_2d: MDS clássico (PCoA).
    """
    codigos  = sorted(str(c) for c in treinos.keys())
    treinos_ = {str(c): treinos[c] for c in treinos}

    dist_mat = _build_cosine_matrix(treinos_, codigos, n_common=n_common)

    best_labels, best_k, best_score, scores_k = _hierarchical_cluster(
        dist_mat, codigos, k_min, k_max
    )

    emb = _classical_mds(dist_mat, n_components=2)

    return _make_resultado(
        labels=best_labels, codigos=codigos, k=best_k,
        silhouette=best_score, scores_k=scores_k,
        features_mat=dist_mat, embedding=emb,
        metodo="cosseno", dist_matrix=dist_mat,
    )


# Método 4 · DTW (Dynamic Time Warping) + hierárquico

def _dtw_distance(
    x      : np.ndarray,
    y      : np.ndarray,
    window : int | None = None,
) -> float:
    """
    Distância DTW entre dois vetores 1D com banda de Sakoe-Chiba.

    Implementação em numpy puro (sem dependências externas).
    O custo acumulado usa distância Euclidiana ponto a ponto (L2).

    Parameters
    ----------
    x, y   : vetores z-score normalizados
    window : largura da banda de Sakoe-Chiba;
             None → sem restrição (DTW clássico)

    Returns
    -------
    float — distância DTW (raiz do custo acumulado mínimo)
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0

    # Garante que a banda cubra a diferença de comprimentos
    if window is None:
        w = max(n, m)
    else:
        w = max(window, abs(n - m))

    # Tabela de programação dinâmica
    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m + 1, i + w + 1)
        for j in range(j_lo, j_hi):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw_mat[i, j] = cost + min(
                dtw_mat[i - 1, j],
                dtw_mat[i, j - 1],
                dtw_mat[i - 1, j - 1],
            )

    return float(np.sqrt(dtw_mat[n, m]))


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score normalização; retorna zeros se std < 1e-9."""
    v   = np.asarray(v, dtype=float)
    mu  = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 1.0
    if std < 1e-9:
        return np.zeros_like(v)
    return (v - mu) / std


def _build_dtw_matrix(
    treinos     : dict,
    codigos     : list,
    window_frac : float = 0.10,
    window_min  : int   = 3,
) -> np.ndarray:
    """
    Matriz de distâncias DTW N×N, normalizada para [0, 1].

    Protocolo anti-vazamento: usa exclusivamente a janela de treino.
    Cada série é z-score normalizada antes do cálculo (compara formas).
    Banda de Sakoe-Chiba: max(window_min, int(window_frac × len_max_do_par)).

    Returns
    -------
    dist_mat : ndarray (N, N), simétrica, diagonal zero, valores em [0, 1]
    """
    n           = len(codigos)
    dist_mat    = np.zeros((n, n), dtype=float)

    # Pré-computa séries z-score
    series_z = {cod: _zscore(np.asarray(treinos[cod], dtype=float))
                for cod in codigos}

    for i in range(n):
        xi = series_z[codigos[i]]
        for j in range(i + 1, n):
            xj  = series_z[codigos[j]]
            w   = max(window_min, int(window_frac * max(len(xi), len(xj))))
            d   = _dtw_distance(xi, xj, window=w)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    np.fill_diagonal(dist_mat, 0.0)

    # Normaliza para [0, 1] pelo valor máximo
    d_max = float(dist_mat.max())
    if d_max > 1e-12:
        dist_mat = dist_mat / d_max

    return dist_mat


def clusterizar_por_dtw(
    treinos      : dict,
    k_min        : int   = 2,
    k_max        : int   = 6,
    window_frac  : float = 0.10,
    window_min   : int   = 3,
    random_state : int   = 42,
) -> ResultadoClustering:
    """Clusteriza por DTW: cada série em z-score (compara forma, não magnitude),
    distância DTW com banda Sakoe-Chiba e hierárquico UPGMA. O DTW tolera
    pequenos deslocamentos temporais (picos em meses vizinhos viram similares);
    em séries fiscais, com calendário alinhado, o ganho é moderado. O
    random_state é ignorado (UPGMA é determinístico), mantido só por uniformidade."""
    codigos  = sorted(str(c) for c in treinos.keys())
    treinos_ = {str(c): treinos[c] for c in treinos}

    logger.info("Construindo matriz DTW para %d series (window_frac=%.2f)...",
                len(codigos), window_frac)

    dist_mat = _build_dtw_matrix(
        treinos_, codigos,
        window_frac=window_frac,
        window_min=window_min,
    )

    best_labels, best_k, best_score, scores_k = _hierarchical_cluster(
        dist_mat, codigos, k_min, k_max
    )

    emb = _classical_mds(dist_mat, n_components=2)

    return _make_resultado(
        labels=best_labels, codigos=codigos, k=best_k,
        silhouette=best_score, scores_k=scores_k,
        features_mat=dist_mat, embedding=emb,
        metodo="dtw", dist_matrix=dist_mat,
    )


# Comparador

def comparar_clusterings(
    treinos      : dict,
    features_df  : Optional[pd.DataFrame] = None,
    k_min        : int   = 2,
    k_max        : int   = 6,
    n_common     : int   = 36,
    min_overlap  : int   = 6,
    window_frac  : float = 0.10,
    window_min   : int   = 3,
    random_state : int   = 42,
) -> Dict[str, ResultadoClustering]:
    """Roda correlacao, cosseno, dtw (e features, se `features_df` vier) e devolve
    {metodo: ResultadoClustering}. Útil na análise exploratória para comparar."""
    resultados: Dict[str, ResultadoClustering] = {}

    logger.info("Comparando clusterings: correlacao, cosseno, dtw%s",
                ", features" if features_df is not None else "")

    resultados["correlacao"] = clusterizar_por_correlacao(
        treinos, k_min=k_min, k_max=k_max,
        min_overlap=min_overlap, random_state=random_state,
    )

    resultados["cosseno"] = clusterizar_por_cosseno(
        treinos, k_min=k_min, k_max=k_max,
        n_common=n_common, random_state=random_state,
    )

    resultados["dtw"] = clusterizar_por_dtw(
        treinos, k_min=k_min, k_max=k_max,
        window_frac=window_frac, window_min=window_min,
        random_state=random_state,
    )

    if features_df is not None:
        resultados["features"] = clusterizar_series(
            features_df, k_min=k_min, k_max=k_max, random_state=random_state,
        )

    # Log resumo
    print("\nComparacao de metodos de clustering:")
    print(f"{'Metodo':<14} {'k':>3}  {'Silhouette':>11}  {'Maior cluster':>14}")
    print("-" * 50)
    for nome, res in sorted(resultados.items(),
                            key=lambda x: x[1].silhouette, reverse=True):
        maior_cl = max(len(v) for v in res.cluster_members.values())
        print(f"  {nome:<12} {res.k:>3}  {res.silhouette:>11.4f}  {maior_cl:>14}")

    return resultados


# Entry point unificado

def clusterizar(
    treinos      : dict,
    features_df  : Optional[pd.DataFrame] = None,
    metodo       : str   = "auto",
    k_min        : int   = 2,
    k_max        : int   = 6,
    n_common     : int   = 36,
    min_overlap  : int   = 6,
    window_frac  : float = 0.10,
    window_min   : int   = 3,
    random_state : int   = 42,
) -> ResultadoClustering:
    """Ponto de entrada do clustering. Escolhe o método em
    {features, correlacao, cosseno, dtw, auto} e devolve o ResultadoClustering —
    no 'auto', o de maior silhouette. O 'features' (e o 'auto') exige features_df."""
    if metodo == "features":
        if features_df is None:
            raise ValueError("features_df e obrigatorio para metodo='features'.")
        return clusterizar_series(
            features_df, k_min=k_min, k_max=k_max, random_state=random_state
        )

    if metodo == "correlacao":
        return clusterizar_por_correlacao(
            treinos, k_min=k_min, k_max=k_max,
            min_overlap=min_overlap, random_state=random_state,
        )

    if metodo == "cosseno":
        return clusterizar_por_cosseno(
            treinos, k_min=k_min, k_max=k_max,
            n_common=n_common, random_state=random_state,
        )

    if metodo == "dtw":
        return clusterizar_por_dtw(
            treinos, k_min=k_min, k_max=k_max,
            window_frac=window_frac, window_min=window_min,
            random_state=random_state,
        )

    if metodo == "auto":
        todos = comparar_clusterings(
            treinos,
            features_df   = features_df,
            k_min         = k_min,
            k_max         = k_max,
            n_common      = n_common,
            min_overlap   = min_overlap,
            window_frac   = window_frac,
            window_min    = window_min,
            random_state  = random_state,
        )
        melhor_nome = max(todos, key=lambda m: todos[m].silhouette)
        melhor      = todos[melhor_nome]
        logger.info("Metodo selecionado: %s (silhouette=%.4f)",
                    melhor_nome, melhor.silhouette)
        print(f"\nMetodo selecionado: {melhor_nome} "
              f"(silhouette={melhor.silhouette:.4f})")
        return melhor

    raise ValueError(
        f"metodo='{metodo}' invalido. "
        "Opcoes: 'features', 'correlacao', 'cosseno', 'dtw', 'auto'."
    )
