"""Experimento 01: gera a base sintética e a grava em dados/sintetico/.

Uso:
    python experimentos/01_gerar_sintetico.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sintetico import gerar_base_sintetica
from src.utils import carregar_config, fixar_sementes, salvar_parquet


def main() -> None:
    config = carregar_config()
    fixar_sementes(config["semente"])
    rng = np.random.default_rng(config["semente"])

    tabelas = gerar_base_sintetica(config, rng)
    destino = config["caminhos"]["dados_sintetico"]
    for nome, df in tabelas.items():
        caminho = salvar_parquet(df, f"{destino}/{nome}.parquet")
        print(f"  {nome:10s}: {len(df):>7d} linhas  -> {caminho}")

    n_taskers = tabelas["taskers"]["tasker_id"].nunique()
    n_ativados = tabelas["taskers"].query("state == 'enabled'")["tasker_id"].nunique()
    print(f"\nProfissionais distintos: {n_taskers} | ativados: {n_ativados} "
          f"({n_ativados / n_taskers:.1%})")


if __name__ == "__main__":
    main()
