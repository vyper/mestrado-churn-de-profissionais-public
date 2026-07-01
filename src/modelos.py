"""Modelos de sobrevivência e seus espaços de busca (Seção 3 da Entrega 3).

Três modelos cobrem do clássico interpretável ao aprendizado de máquina
não-linear, na linha de Puram et al. (2026):

- Cox PH regularizado (L2 / ridge), linha de base interpretável e sensível à
  escala (usa o pré-processador com padronização);
- Random Survival Forest (RSF), vencedor em Puram et al. (2026);
- Gradient Boosting de sobrevivência (perda de Cox), família CoxBoost.

Cada fábrica devolve `(estimador, distribuicao_de_parametros, escalar)`, onde
`distribuicao_de_parametros` usa o prefixo `modelo__` para compor com o
`Pipeline(preprocessador, modelo)`.
"""
from __future__ import annotations

import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis


def construir_alvo_estruturado(tempo, evento) -> np.ndarray:
    """Converte (tempo, evento) no array estruturado exigido pelo scikit-survival."""
    return np.array(list(zip(np.asarray(evento, dtype=bool), np.asarray(tempo, dtype=float))),
                    dtype=[("evento", "?"), ("tempo", "<f8")])


def modelo_cox(config: dict):
    """Cox PH com penalização ridge (L2); o `alpha` é varrido em escala log."""
    est = CoxPHSurvivalAnalysis(alpha=1.0)
    espaco = {"modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return est, espaco, True  # escalar=True


def modelo_rsf(config: dict, semente: int):
    """Random Survival Forest.

    n_jobs=1 aqui: a paralelização fica a cargo do RandomizedSearchCV (n_jobs=-1),
    que distribui os candidatos entre os núcleos. Deixar ambos com n_jobs=-1
    causaria paralelismo aninhado (oversubscription) e degradaria o desempenho.
    """
    par = config["modelos"]["rsf"]
    est = RandomSurvivalForest(random_state=semente, n_jobs=1)
    espaco = {
        "modelo__n_estimators": par["n_estimators"],
        "modelo__min_samples_leaf": par["min_samples_leaf"],
        "modelo__max_features": par["max_features"],
        "modelo__max_depth": par["max_depth"],
    }
    return est, espaco, False


def modelo_gbm(config: dict, semente: int):
    """Gradient Boosting de sobrevivência (perda de Cox)."""
    par = config["modelos"]["gbm"]
    est = GradientBoostingSurvivalAnalysis(random_state=semente)
    espaco = {
        "modelo__n_estimators": par["n_estimators"],
        "modelo__learning_rate": par["learning_rate"],
        "modelo__max_depth": par["max_depth"],
        "modelo__subsample": par["subsample"],
    }
    return est, espaco, False


def fabricas_modelos(config: dict, semente: int) -> dict:
    """Mapa nome -> (estimador, espaço de busca, escalar) dos três modelos."""
    return {
        "Cox PH": modelo_cox(config),
        "RSF": modelo_rsf(config, semente),
        "Gradient Boosting": modelo_gbm(config, semente),
    }
