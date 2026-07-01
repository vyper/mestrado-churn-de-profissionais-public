"""Experimento 04: comparação estatística formal entre os modelos.

Lê resultados/tabelas/metricas_por_fold.csv (gerado no experimento 03), aplica
Friedman + Nemenyi sobre o C-index de Uno e exporta a tabela do teste e o
diagrama de diferença crítica.

Uso:
    python experimentos/04_comparar_estatistica.py [--metrica c_uno]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.estatistica import comparar
from src.utils import (carregar_config, caminho_abs, caminho_figura,
                       exportar_tabela_latex)
from src.visualizacao import diagrama_diferenca_critica


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica", default="c_uno",
                    choices=["c_uno", "c_harrell", "auc_media", "ibs"])
    args = ap.parse_args()

    config = carregar_config()
    dir_tab = caminho_abs(config["caminhos"]["resultados_tabelas"])
    por_fold = pd.read_csv(dir_tab / "metricas_por_fold.csv")

    res = comparar(por_fold, metrica=args.metrica)
    fr = res["friedman"]
    print(f"Friedman ({args.metrica}): estatística={fr['estatistica']:.3f} "
          f"p-valor={fr['p_valor']:.4f}")
    print("\nRanks médios (1 = melhor):")
    print(res["ranks_medios"].to_string())
    print(f"\nDistância crítica (Nemenyi, alpha=0.05): {res['distancia_critica']:.3f}")
    print("\nNemenyi (p-valores pareados):")
    print(res["nemenyi"].round(4).to_string())

    # Exporta tabela do teste e p-valores de Nemenyi.
    tab_friedman = pd.DataFrame([{
        "metrica": args.metrica,
        "friedman_estat": round(fr["estatistica"], 4),
        "friedman_p": round(fr["p_valor"], 4),
        "distancia_critica": round(res["distancia_critica"], 4),
    }])
    exportar_tabela_latex(tab_friedman, "teste_friedman", config,
                          caption="Teste de Friedman entre os modelos.",
                          label="tab:friedman")
    nemenyi = res["nemenyi"].round(4).reset_index().rename(columns={"index": "modelo"})
    exportar_tabela_latex(nemenyi, "posthoc_nemenyi", config,
                          caption="Post-hoc de Nemenyi (p-valores pareados).",
                          label="tab:nemenyi")
    diagrama_diferenca_critica(res["ranks_medios"], res["distancia_critica"],
                               caminho_figura("diferenca_critica.pdf", config))
    print("\nTabelas e diagrama de diferença crítica exportados.")


if __name__ == "__main__":
    main()
