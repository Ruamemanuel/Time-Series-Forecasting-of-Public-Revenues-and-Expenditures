"""
Clusteriza as séries por similaridade de features (K-Means). As features saem só
da janela de treino — o conjunto de teste não entra. O k é buscado em [k_min, k_max] pelo
maior silhouette, com as features em z-score antes do K-Means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ResultadoClustering:
    """Resultado completo do processo de clustering."""
    labels          : np.ndarray        # label por serie (ordem = features_df.index)
    codigos         : list              # codigos na mesma ordem das labels
    k               : int               # numero de clusters escolhido
    silhouette      : float             # silhouette score do melhor k
    features_scaled : np.ndarray        # features z-score (usadas no KMeans)
    cluster_map     : dict              # {codigo: cluster_id}
    cluster_members : dict              # {cluster_id: [codigos]}
    silhouette_per_k: dict = field(default_factory=dict)  # {k: score}


def clusterizar_series(
    features_df: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = 42,
    n_init: int = 20,
) -> ResultadoClustering:
    """Agrupa as séries por similaridade de features (K-Means com busca de k).
    Recebe o DataFrame de features indexado por código e devolve o
    ResultadoClustering."""
    n = len(features_df)
    k_max_real = min(k_max, n - 1)

    if k_max_real < k_min:
        # poucos dados: tudo num cluster só
        labels = np.zeros(n, dtype=int)
        cluster_map = {cod: 0 for cod in features_df.index}
        cluster_members = {0: list(features_df.index)}
        X = np.zeros((n, features_df.shape[1]))
        return ResultadoClustering(
            labels=labels, codigos=list(features_df.index),
            k=1, silhouette=0.0, features_scaled=X,
            cluster_map=cluster_map, cluster_members=cluster_members,
        )

    codigos = list(features_df.index)
    scaler  = StandardScaler()
    X       = scaler.fit_transform(features_df.fillna(0.0).values)

    best_k      = k_min
    best_score  = -1.0
    best_labels = None
    scores_k    = {}

    for k in range(k_min, k_max_real + 1):
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=500,
        )
        labels = km.fit_predict(X)

        # Silhouette requer ao menos 2 clusters distintos
        if len(np.unique(labels)) < 2:
            continue

        score       = float(silhouette_score(X, labels))
        scores_k[k] = score
        logger.debug("k=%d  silhouette=%.4f", k, score)

        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = labels.copy()

    if best_labels is None:
        km          = KMeans(n_clusters=k_min, random_state=random_state, n_init=n_init)
        best_labels = km.fit_predict(X)
        best_k      = k_min
        best_score  = -1.0

    cluster_map     = {cod: int(lbl) for cod, lbl in zip(codigos, best_labels)}
    cluster_members : dict = {}
    for cod, lbl in cluster_map.items():
        cluster_members.setdefault(lbl, []).append(cod)

    sizes = {k: len(v) for k, v in cluster_members.items()}
    logger.info(
        "Clustering: k=%d (silhouette=%.4f) | series por cluster: %s",
        best_k, best_score, sizes,
    )

    return ResultadoClustering(
        labels           = best_labels,
        codigos          = codigos,
        k                = best_k,
        silhouette       = best_score,
        features_scaled  = X,
        cluster_map      = cluster_map,
        cluster_members  = cluster_members,
        silhouette_per_k = scores_k,
    )
