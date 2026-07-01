"""Experimento 02: constrói a cohort e a matriz de atributos a partir das tabelas.

Lê de dados/sintetico/ (ou dados/bruto/ quando --fonte bruto) e grava a cohort
final em dados/processado/cohort.parquet.

Uso:
    python experimentos/02_construir_cohort.py [--fonte sintetico|bruto]
"""
from __future__ import annotations

import os

# Capa as threads BLAS/Accelerate para não saturar a máquina na construção da cohort.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.atributos import construir_atributos
from src.cohort import montar_cohort, resumo_cohort
from src.utils import (carregar_config, carregar_parquet, caminho_figura,
                       exportar_tabela_latex, salvar_parquet)
from src.visualizacao import curva_kaplan_meier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", choices=["sintetico", "bruto"], default="sintetico")
    args = ap.parse_args()

    config = carregar_config()
    base = config["caminhos"][f"dados_{args.fonte}"]
    taskers = carregar_parquet(f"{base}/taskers.parquet")
    jobs = carregar_parquet(f"{base}/jobs.parquet")
    incidents = carregar_parquet(f"{base}/incidents.parquet")

    cohort = montar_cohort(taskers, jobs, incidents, config)
    matriz = construir_atributos(cohort, jobs, incidents)

    caminho = salvar_parquet(matriz, f"{config['caminhos']['dados_processado']}/cohort.parquet")
    resumo = resumo_cohort(cohort)
    print(f"Cohort gravada em {caminho}")
    print(f"  profissionais : {resumo['n_profissionais']}")
    print(f"  eventos       : {resumo['n_eventos']} ({resumo['taxa_evento']:.1%})")
    print(f"  censurados    : {resumo['n_censurados']} ({1 - resumo['taxa_evento']:.1%})")
    print(f"  tempo mediano : {resumo['tempo_mediano']:.0f} dias | máx {resumo['tempo_max']:.0f}")
    print(f"  atributos     : {matriz.shape[1] - 3} colunas preditoras")

    # Tabela descritiva da cohort e curva de Kaplan-Meier global (artefatos do artigo).
    # Valores formatados como texto (contagens inteiras, taxas em %) para o LaTeX.
    descritiva = pd.DataFrame([
        {"indicador": "Profissionais", "valor": f"{resumo['n_profissionais']:d}"},
        {"indicador": "Eventos (churn)", "valor": f"{resumo['n_eventos']:d}"},
        {"indicador": "Censurados", "valor": f"{resumo['n_censurados']:d}"},
        {"indicador": "Taxa de evento", "valor": f"{resumo['taxa_evento']:.1%}"},
        {"indicador": "Tempo mediano (dias)", "valor": f"{resumo['tempo_mediano']:.0f}"},
        {"indicador": "Tempo máximo (dias)", "valor": f"{resumo['tempo_max']:.0f}"},
        {"indicador": "Atributos preditores", "valor": f"{matriz.shape[1] - 3:d}"},
    ])
    exportar_tabela_latex(descritiva, "descritiva_cohort", config,
                          caption="Estatísticas descritivas da cohort final.",
                          label="tab:descritiva")
    curva_kaplan_meier(matriz["tempo"], matriz["evento"],
                       caminho_figura("kaplan_meier.pdf", config),
                       titulo="Sobrevivência (permanência) da cohort")
    print("  artefatos     : descritiva_cohort.tex e kaplan_meier.pdf exportados")


if __name__ == "__main__":
    main()
