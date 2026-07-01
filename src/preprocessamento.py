"""Pipelines de pré-processamento, ajustados apenas no treino de cada partição.

Implementa a Seção 2 da Entrega 3. Há duas variantes, conforme o modelo:

- `escalar=True`  (Cox PH): imputação + indicador de ausência, one-hot e
  padronização z-score (o Cox é sensível à escala e à regularização).
- `escalar=False` (RSF / Gradient Boosting): mesma imputação e codificação,
  sem padronização (árvores são invariantes a transformações monotônicas).

O encapsulamento em `ColumnTransformer`/`Pipeline` garante que imputação,
codificação e padronização sejam aprendidas só no treino, evitando o vazamento
discutido por Kimura (2022) e AbdelAziz et al. (2025).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Atributos categóricos de baixa cardinalidade (one-hot).
COLS_CATEGORICAS = ["genero", "plano", "canal_indicacao"]

# Colunas que nunca entram como preditor.
COLS_EXCLUIR = ["tasker_id", "tempo", "evento"]


def colunas_preditoras(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa as colunas preditoras em (numéricas, categóricas)."""
    preditoras = [c for c in df.columns if c not in COLS_EXCLUIR]
    categoricas = [c for c in preditoras if c in COLS_CATEGORICAS]
    numericas = [c for c in preditoras if c not in COLS_CATEGORICAS]
    return numericas, categoricas


def construir_preprocessador(df: pd.DataFrame, escalar: bool) -> ColumnTransformer:
    """Monta o ColumnTransformer de pré-processamento para a matriz de atributos.

    `add_indicator=True` na imputação numérica preserva o sinal de ausência
    informativa (reputação de recém-ativados), conforme a Seção 2 da Entrega 3.
    """
    numericas, categoricas = colunas_preditoras(df)

    passos_num = [("imputacao", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        passos_num.append(("escala", StandardScaler()))
    pipe_num = Pipeline(passos_num)

    pipe_cat = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", pipe_num, numericas),
        ("cat", pipe_cat, categoricas),
    ], remainder="drop")
