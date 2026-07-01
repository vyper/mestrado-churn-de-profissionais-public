"""Métricas de avaliação sensíveis ao tempo (Seção 4 da Entrega 3).

Reúne C-index de Harrell e de Uno (IPCW), AUC dependente do tempo, Brier score
e sua versão integrada (IBS). Esse conjunto avalia tanto o ordenamento de risco
quanto a qualidade probabilística/temporal e é robusto ao desbalanceamento de
eventos, evitando a armadilha da acurácia (Kimura, 2022).

Para que a ponderação IPCW do scikit-survival permaneça bem definida, a
avaliação é feita até um horizonte `tau` e os conjuntos são censurados
administrativamente nesse horizonte (observações além de `tau` viram censuras
em `tau`). É a prática usual em avaliação dependente do tempo e evita o erro
"censoring survival function is zero".
"""
from __future__ import annotations

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


def _max_censurado(y: np.ndarray) -> float:
    """Maior tempo censurado (onde a função de censura IPCW ainda é positiva)."""
    censurados = y["tempo"][~y["evento"]]
    return float(censurados.max()) if len(censurados) else float(y["tempo"].max())


def tempos_avaliacao(y_train: np.ndarray, y_test: np.ndarray, quantis) -> tuple[np.ndarray, float]:
    """Retorna (tempos de avaliação, horizonte tau) válidos para AUC(t)/Brier(t).

    Os tempos ficam no suporte comum a treino e teste e estritamente abaixo de
    `tau`, definido como o menor entre os maiores tempos censurados de ambos os
    conjuntos. A censura administrativa em `tau` (ver `_censurar_em`) garante que
    a função de censura não se anule nos pontos avaliados.
    """
    t_train_evt = y_train["tempo"][y_train["evento"]]
    t_test_evt = y_test["tempo"][y_test["evento"]]
    if len(t_train_evt) == 0 or len(t_test_evt) == 0:
        raise ValueError("Sem eventos suficientes para definir tempos de avaliação.")
    limite_inf = max(t_train_evt.min(), t_test_evt.min())
    tau = min(_max_censurado(y_train), _max_censurado(y_test),
              t_train_evt.max(), t_test_evt.max())
    if tau <= limite_inf:
        raise ValueError("Suporte temporal comum insuficiente para a avaliação.")
    # Tempos estritamente internos a (limite_inf, tau).
    limite_sup = limite_inf + 0.95 * (tau - limite_inf)
    internos = t_test_evt[(t_test_evt > limite_inf) & (t_test_evt < limite_sup)]
    tempos = np.quantile(internos, quantis) if len(internos) else \
        np.linspace(limite_inf, limite_sup, len(quantis) + 2)[1:-1]
    tempos = np.unique(np.clip(tempos, limite_inf + 1e-3, limite_sup))
    return tempos, float(tau)


def _censurar_em(y: np.ndarray, tau: float) -> np.ndarray:
    """Censura administrativamente em `tau`: observações >= tau viram censuras em tau.

    Usar `>=` garante que a maior observação resultante seja sempre uma censura,
    condição para a função de censura IPCW permanecer positiva nos tempos de
    evento avaliados (todos estritamente menores que `tau`).
    """
    z = y.copy()
    alem = z["tempo"] >= tau
    z["evento"][alem] = False
    z["tempo"][alem] = tau
    return z


def avaliar(modelo, X_test, y_train: np.ndarray, y_test: np.ndarray,
            tempos: np.ndarray, tau: float) -> dict[str, float]:
    """Calcula as métricas para um modelo já ajustado, no conjunto de teste."""
    y_tr = _censurar_em(y_train, tau)
    y_te = _censurar_em(y_test, tau)
    risco = modelo.predict(X_test)  # escore de risco (maior = mais risco)

    c_harrell = concordance_index_censored(y_te["evento"], y_te["tempo"], risco)[0]
    c_uno = concordance_index_ipcw(y_tr, y_te, risco, tau=tau)[0]
    _, auc_media = cumulative_dynamic_auc(y_tr, y_te, risco, tempos)

    # Brier integrado (IBS): exige funções de sobrevivência preditas nos `tempos`.
    surv_fns = modelo.predict_survival_function(X_test)
    surv_probs = np.row_stack([fn(tempos) for fn in surv_fns])
    ibs = integrated_brier_score(y_tr, y_te, surv_probs, tempos)

    return {
        "c_harrell": float(c_harrell),
        "c_uno": float(c_uno),
        "auc_media": float(auc_media),
        "ibs": float(ibs),
    }
