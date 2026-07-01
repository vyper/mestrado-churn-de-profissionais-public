"""Experimento 06: análises de sensibilidade do limiar W e do landmark.

Reconstrói a cohort para cada combinação de W (30/60/90) e landmark (14/30/45),
reportando a estabilidade das taxas de evento, do tempo mediano e do
ordenamento dos modelos pelo C-index de Uno. Exporta a tabela de sensibilidade.

Uso:
    python experimentos/06_sensibilidade.py [--fonte sintetico|bruto] [--sem-ranking]
"""
from __future__ import annotations

import os

# Capa as threads BLAS/OpenMP/Accelerate por worker ANTES de importar numpy/sklearn
# (config: paralelismo.n_jobs), mantendo a máquina responsiva na varredura.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensibilidade import analisar_sensibilidade
from src.utils import (carregar_config, carregar_parquet, exportar_tabela_latex,
                       fixar_sementes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", choices=["sintetico", "bruto"], default="sintetico")
    ap.add_argument("--sem-ranking", action="store_true",
                    help="reporta só estatísticas da cohort, sem reavaliar modelos")
    args = ap.parse_args()

    config = carregar_config()
    fixar_sementes(config["semente"])
    base = config["caminhos"][f"dados_{args.fonte}"]
    taskers = carregar_parquet(f"{base}/taskers.parquet")
    jobs = carregar_parquet(f"{base}/jobs.parquet")
    incidents = carregar_parquet(f"{base}/incidents.parquet")

    tabela = analisar_sensibilidade(taskers, jobs, incidents, config,
                                    avaliar_ranking=not args.sem_ranking)
    exportar_tabela_latex(tabela.round(4), "sensibilidade", config,
                          caption="Análise de sensibilidade (W e landmark).",
                          label="tab:sensibilidade")
    print(tabela.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
