"""Análises de sensibilidade das principais decisões operacionais (Seção 4).

Varia o limiar de inatividade W (30/60/90) e o marco de landmarking (14/30/45),
reconstruindo a cohort e reavaliando, para verificar a estabilidade das taxas
de evento, das curvas de sobrevivência e do ORDENAMENTO dos modelos.

Para manter o custo computacional baixo, o ordenamento é estimado com modelos
de hiperparâmetros fixos e modestos (sem busca) em validação cruzada de 3
folds; o objetivo aqui é a estabilidade relativa, não o desempenho absoluto.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.atributos import construir_atributos
from src.avaliacao import avaliar, tempos_avaliacao
from src.cohort import montar_cohort, resumo_cohort
from src.modelos import construir_alvo_estruturado
from src.preprocessamento import construir_preprocessador
from src.utils import n_jobs


def _modelos_rapidos(semente: int, nucleos: int = -1) -> dict:
    """Modelos de configuração fixa e modesta, para comparação rápida de ranking."""
    return {
        "Cox PH": (CoxPHSurvivalAnalysis(alpha=1.0), True),
        "RSF": (RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                     random_state=semente, n_jobs=nucleos), False),
        "Gradient Boosting": (GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=semente), False),
    }


def _c_uno_medio(matriz: pd.DataFrame, config: dict, semente: int) -> dict[str, float]:
    """C-index de Uno médio (3 folds) por modelo, sem busca de hiperparâmetros."""
    X = matriz.drop(columns=["tempo", "evento"])
    y = construir_alvo_estruturado(matriz["tempo"], matriz["evento"])
    evento = matriz["evento"].to_numpy()
    quantis = config["protocolo"]["tempos_avaliacao_quantis"]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=semente)

    nucleos = n_jobs(config)
    modelos = _modelos_rapidos(semente, nucleos)
    resultado: dict[str, list[float]] = {nome: [] for nome in modelos}
    for nome, (est, escalar) in modelos.items():
        for idx_tr, idx_te in skf.split(X, evento):
            pre = construir_preprocessador(X.iloc[idx_tr], escalar=escalar)
            pipe = Pipeline([("pre", pre), ("modelo", est)])
            pipe.fit(X.iloc[idx_tr], y[idx_tr])
            tempos, tau = tempos_avaliacao(y[idx_tr], y[idx_te], quantis)
            m = avaliar(pipe, X.iloc[idx_te], y[idx_tr], y[idx_te], tempos, tau)
            resultado[nome].append(m["c_uno"])
    return {nome: float(np.mean(vs)) for nome, vs in resultado.items()}


def analisar_sensibilidade(taskers, jobs, incidents, config: dict,
                           avaliar_ranking: bool = True) -> pd.DataFrame:
    """Varre as grades de W e landmark, reconstruindo a cohort para cada combinação."""
    semente = config["semente"]
    grade_W = config["sensibilidade"]["janela_inatividade_dias"]
    grade_lm = config["sensibilidade"]["landmark_dias"]

    linhas = []
    for W in grade_W:
        for lm in grade_lm:
            cfg = copy.deepcopy(config)
            cfg["cohort"]["janela_inatividade_dias"] = W
            cfg["cohort"]["landmark_dias"] = lm
            cohort = montar_cohort(taskers, jobs, incidents, cfg)
            resumo = resumo_cohort(cohort)
            linha = {"W": W, "landmark": lm, **resumo}
            if avaliar_ranking and resumo["n_eventos"] > 30:
                matriz = construir_atributos(cohort, jobs, incidents)
                for nome, c in _c_uno_medio(matriz, cfg, semente).items():
                    linha[f"c_uno[{nome}]"] = c
            linhas.append(linha)
    return pd.DataFrame(linhas)
