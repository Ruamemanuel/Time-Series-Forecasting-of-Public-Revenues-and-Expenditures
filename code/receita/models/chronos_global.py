"""
Inferência zero-shot com o Chronos (amazon/chronos-t5-small) no paradigma global.
Não há treino nos dados do projeto: o modelo pré-treinado é carregado uma vez e
aplicado direto a cada série, prevendo de forma recursiva. É o uso típico de um
foundation model como baseline global.
"""

from __future__ import annotations

import numpy as np
import torch


def carregar_pipeline(model_id: str = "amazon/chronos-t5-small"):
    """Carrega o pipeline Chronos (em CPU, para máxima compatibilidade)."""
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return pipeline


def prever_rolling_chronos(
    pipeline,
    treino: np.ndarray | list,
    teste: np.ndarray | list,
    num_samples: int = 20,
    context_length: int = 12,
) -> np.ndarray:
    """Previsão recursiva com o Chronos (zero-shot).

    A cada passo o contexto é o histórico acumulado e o modelo prevê 1 ponto; a
    previsão (não o valor real) realimenta o histórico, igual aos demais modelos.
    Assim a comparação é justa: todos projetam o horizonte recursivamente, sem
    ver o conjunto de teste. O `context_length` é fixado em N_LAGS_GLOBAL (12) para casar
    com os globais sklearn/LGBM. `teste` só define quantos passos prever; devolve
    um array com len(teste) previsões na escala do input."""
    treino = np.asarray(treino, dtype=np.float32)
    teste  = np.asarray(teste,  dtype=np.float32)

    historico = list(treino)
    preds = []
    for _ in range(len(teste)):
        # Limita ao mesmo número de lags usado pelos modelos globais sklearn/LGBM
        contexto = np.asarray(historico[-context_length:], dtype=np.float32)
        ctx_tensor = torch.tensor(contexto).unsqueeze(0)  # shape (1, T)

        with torch.no_grad():
            forecast = pipeline.predict(
                ctx_tensor,
                prediction_length=1,
                num_samples=num_samples,
            )
        # forecast shape: (1, num_samples, 1)
        mediana = float(np.median(forecast[0, :, 0].numpy()))
        preds.append(mediana)
        historico.append(mediana)   # multi-step: alimenta com a PREVISAO

    return np.array(preds, dtype=float)
