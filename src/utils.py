"""Utilidades transversais: configuração, sementes, IO e registro de execuções.

Centraliza as decisões de reprodutibilidade da Seção 4 da Entrega 3:
fixação de sementes, IO padronizado em parquet e registro das configurações
de cada execução.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Raiz do repositório (dois níveis acima deste arquivo: src/ -> raiz).
RAIZ = Path(__file__).resolve().parent.parent


def carregar_config(caminho: str | Path | None = None) -> dict[str, Any]:
    """Carrega o config.yaml como dicionário."""
    if caminho is None:
        caminho = RAIZ / "config" / "config.yaml"
    with open(caminho, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fixar_sementes(semente: int) -> None:
    """Fixa as sementes de todas as fontes de aleatoriedade usadas no projeto."""
    random.seed(semente)
    np.random.seed(semente)
    os.environ["PYTHONHASHSEED"] = str(semente)


def n_jobs(config: dict[str, Any]) -> int:
    """Número de núcleos para paralelização (busca/CV, permutação, floresta).

    Lê `paralelismo.n_jobs` do config:
      - "auto": usa (núcleos_totais - `paralelismo.nucleos_livres`), com piso 1,
        deixando núcleos livres para o sistema (evita travar a máquina);
      - inteiro: usa o valor fixo (por ex. 6), ou -1 para todos os núcleos.
    """
    par = config.get("paralelismo", {})
    val = par.get("n_jobs", "auto")
    if isinstance(val, str) and val.strip().lower() == "auto":
        livres = int(par.get("nucleos_livres", 2))
        total = os.cpu_count() or 2
        return max(1, total - livres)
    return int(val)


def caminho_abs(rel: str | Path) -> Path:
    """Resolve um caminho relativo (do config) contra a raiz do repositório."""
    p = Path(rel)
    return p if p.is_absolute() else RAIZ / p


def salvar_parquet(df: pd.DataFrame, rel: str | Path) -> Path:
    """Salva um DataFrame em parquet, criando o diretório se necessário."""
    destino = caminho_abs(rel)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return destino


def carregar_parquet(rel: str | Path) -> pd.DataFrame:
    """Carrega um parquet a partir de um caminho relativo à raiz."""
    return pd.read_parquet(caminho_abs(rel))


def registrar_execucao(nome: str, config: dict[str, Any], metricas: dict[str, Any] | None = None) -> Path:
    """Registra a configuração e (opcionalmente) métricas de uma execução.

    Cada execução vira um JSON em resultados/execucoes/, com carimbo de tempo,
    sustentando a rastreabilidade exigida na Seção 4 da Entrega 3.
    """
    dir_exec = caminho_abs(config["caminhos"]["resultados_execucoes"])
    dir_exec.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = dir_exec / f"{nome}_{carimbo}.json"
    payload = {"nome": nome, "carimbo": carimbo, "config": config}
    if metricas is not None:
        payload["metricas"] = metricas
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return destino


def exportar_tabela_latex(df: pd.DataFrame, nome: str, config: dict[str, Any], **kwargs: Any) -> Path:
    """Exporta um DataFrame como tabela LaTeX (.tex) e cópia .csv em resultados/tabelas."""
    dir_tab = caminho_abs(config["caminhos"]["resultados_tabelas"])
    dir_tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(dir_tab / f"{nome}.csv", index=False)
    destino = dir_tab / f"{nome}.tex"
    df.to_latex(destino, index=False, float_format="%.4f", **kwargs)
    return destino


def caminho_figura(nome: str, config: dict[str, Any]) -> Path:
    """Caminho de saída para uma figura em resultados/figuras (cria o diretório)."""
    dir_fig = caminho_abs(config["caminhos"]["resultados_figuras"])
    dir_fig.mkdir(parents=True, exist_ok=True)
    return dir_fig / nome
