"""Interpretabilidade sobre o modelo de melhor desempenho (Seção 4 da Entrega 3).

Importância por permutação, dependência parcial (PDP) e, quando disponível,
valores SHAP. Identifica os fatores que mais influenciam o tempo até o churn
(com atenção esperada ao deslocamento e aos sinais de atrito operacional), à
maneira de Puram et al. (2026) e AbdelAziz et al. (2025).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def importancia_permutacao(modelo, X, y, n_repeticoes: int = 10,
                           semente: int = 42, nucleos: int = -1) -> pd.DataFrame:
    """Importância por permutação, pontuada pelo C-index de Harrell do modelo.

    `modelo` é o Pipeline ajustado (pré-processador + estimador); a permutação
    é aplicada às colunas originais de `X`, respeitando o pré-processamento.
    `nucleos` controla a paralelização (padrão -1 = todos; ver config.paralelismo).
    """
    resultado = permutation_importance(modelo, X, y, n_repeats=n_repeticoes,
                                       random_state=semente, n_jobs=nucleos)
    return (pd.DataFrame({
        "atributo": X.columns,
        "importancia_media": resultado.importances_mean,
        "importancia_desvio": resultado.importances_std,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True))


def dependencia_parcial(modelo, X: pd.DataFrame, atributo: str,
                        n_pontos: int = 20) -> pd.DataFrame:
    """Dependência parcial de uma covariável numérica sobre o escore de risco médio."""
    valores = np.linspace(X[atributo].quantile(0.05), X[atributo].quantile(0.95), n_pontos)
    riscos = []
    base = X.copy()
    for v in valores:
        base[atributo] = v
        riscos.append(float(np.mean(modelo.predict(base))))
    return pd.DataFrame({atributo: valores, "risco_medio": riscos})


def shap_valores(modelo, X: pd.DataFrame, n_amostras: int = 100, n_fundo: int = 50,
                 semente: int = 42):
    """Valores SHAP model-agnostic sobre o escore de risco do Pipeline.

    Explica `modelo.predict` (risco) diretamente nas covariáveis originais,
    funcionando para Cox, RSF e Gradient Boosting sem depender do suporte do
    TreeExplainer aos estimadores do scikit-survival. Retorna
    (importancia_media_abs, nomes) ou None se o SHAP não estiver instalado.
    """
    try:
        import shap
    except ImportError:
        return None

    # SHAP no espaço pré-processado (numérico): evita colunas categóricas/strings
    # no masker e funciona para Cox, RSF e GBM sobre o escore de risco.
    pre = modelo.named_steps["pre"]
    est = modelo.named_steps["modelo"]
    nomes = list(pre.get_feature_names_out())
    fundo = pre.transform(X.sample(min(n_fundo, len(X)), random_state=semente))
    amostra = pre.transform(X.sample(min(n_amostras, len(X)), random_state=semente + 1))
    explainer = shap.Explainer(est.predict, fundo)
    expl = explainer(amostra, silent=True)
    importancia = np.abs(expl.values).mean(axis=0)
    return importancia, nomes
