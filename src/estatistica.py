"""Comparação estatística formal entre modelos (Seção 4 da Entrega 3).

Sobre as métricas por partição, aplica o teste de Friedman (diferença global
entre os três modelos) e, havendo significância, o post-hoc de Nemenyi com o
diagrama de diferença crítica. Endereça a lacuna comum apontada na Entrega 2
(ausência de teste estatístico ao comparar modelos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare


def matriz_por_fold(por_fold: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Reorganiza as métricas em uma matriz (linhas = folds, colunas = modelos)."""
    return por_fold.pivot(index="fold", columns="modelo", values=metrica)


def teste_friedman(matriz: pd.DataFrame) -> dict:
    """Teste de Friedman sobre a matriz fold x modelo. Maior métrica = melhor."""
    amostras = [matriz[col].to_numpy() for col in matriz.columns]
    estat, p = friedmanchisquare(*amostras)
    return {"estatistica": float(estat), "p_valor": float(p), "modelos": list(matriz.columns)}


def posthoc_nemenyi(matriz: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc de Nemenyi (p-valores pareados) sobre a matriz fold x modelo."""
    return sp.posthoc_nemenyi_friedman(matriz.to_numpy())


def ranks_medios(matriz: pd.DataFrame, maior_melhor: bool = True) -> pd.Series:
    """Ranks médios por modelo (rank 1 = melhor)."""
    sinal = -1 if maior_melhor else 1
    ranks = (sinal * matriz).rank(axis=1)
    media = ranks.mean(axis=0)
    media.index = matriz.columns
    return media.sort_values()


def distancia_critica(n_modelos: int, n_folds: int, alpha: float = 0.05) -> float:
    """Distância crítica de Nemenyi (CD) para o diagrama de diferença crítica."""
    # Valores críticos q_alpha da estatística do intervalo studentizado / sqrt(2).
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}.get(n_modelos, 2.343)
    return float(q_alpha * np.sqrt(n_modelos * (n_modelos + 1) / (6.0 * n_folds)))


def comparar(por_fold: pd.DataFrame, metrica: str = "c_uno") -> dict:
    """Executa Friedman + Nemenyi + ranks médios + CD para uma métrica."""
    matriz = matriz_por_fold(por_fold, metrica)
    friedman = teste_friedman(matriz)
    nemenyi = posthoc_nemenyi(matriz)
    nemenyi.index = matriz.columns
    nemenyi.columns = matriz.columns
    ranks = ranks_medios(matriz, maior_melhor=(metrica != "ibs"))
    cd = distancia_critica(matriz.shape[1], matriz.shape[0])
    return {
        "metrica": metrica,
        "friedman": friedman,
        "nemenyi": nemenyi,
        "ranks_medios": ranks,
        "distancia_critica": cd,
    }
