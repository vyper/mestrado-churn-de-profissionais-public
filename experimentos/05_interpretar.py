"""Experimento 05: interpretabilidade sobre o modelo de melhor desempenho.

Seleciona o melhor modelo pelo C-index de Uno médio (resultados do experimento
03), recarrega o estimador final salvo e produz: importância por permutação
(tabela + figura), dependência parcial das principais covariáveis e, quando o
SHAP estiver instalado, o resumo SHAP.

Uso:
    python experimentos/05_interpretar.py
"""
from __future__ import annotations

import os

# Capa as threads BLAS/OpenMP/Accelerate por worker ANTES de importar numpy/sklearn,
# para a permutação paralelizar sem oversubscription (config: paralelismo.n_jobs).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interpretabilidade import (dependencia_parcial, importancia_permutacao,
                                     shap_valores)
from src.modelos import construir_alvo_estruturado
from src.utils import (carregar_config, carregar_parquet, caminho_abs,
                       caminho_figura, exportar_tabela_latex, n_jobs)
from src.visualizacao import grafico_importancia, km_estratificado_reputacao


def main() -> None:
    config = carregar_config()
    semente = config["semente"]
    dir_tab = caminho_abs(config["caminhos"]["resultados_tabelas"])
    por_fold = pd.read_csv(dir_tab / "metricas_por_fold.csv")
    melhor = por_fold.groupby("modelo")["c_uno"].mean().idxmax()
    print(f"Melhor modelo (C-Uno médio): {melhor}")

    modelo = joblib.load(caminho_abs(f"resultados/modelos/{melhor.replace(' ', '_')}.joblib"))
    df = carregar_parquet(f"{config['caminhos']['dados_processado']}/cohort.parquet")
    X = df.drop(columns=["tempo", "evento"])
    y = construir_alvo_estruturado(df["tempo"], df["evento"])

    imp = importancia_permutacao(modelo, X, y, n_repeticoes=10, semente=semente,
                                 nucleos=n_jobs(config))
    exportar_tabela_latex(imp.head(15), "importancia_permutacao", config,
                          caption=f"Importância por permutação ({melhor}).",
                          label="tab:importancia")
    grafico_importancia(imp["atributo"].tolist(), imp["importancia_media"].tolist(),
                        caminho_figura("importancia_permutacao.pdf", config),
                        titulo=f"Importância por permutação ({melhor})")
    print("\nTop 10 atributos (importância por permutação):")
    print(imp.head(10).to_string(index=False))

    # Dependência parcial (PDP): as 3 covariáveis numéricas mais importantes,
    # somadas às features de interesse fixadas no config (cluster de reputação),
    # para que os PDPs que sustentam a discussão sejam reprodutíveis pelo pipeline.
    # Remove PDPs de execuções anteriores: o conjunto de top-features muda entre
    # runs, e arquivos antigos sobreviventes geram artefatos inconsistentes.
    for antigo in dir_tab.glob("pdp_*.csv"):
        antigo.unlink()
    top_numericas = [a for a in imp["atributo"] if X[a].dtype.kind in "fi"][:3]
    fixas = config.get("interpretabilidade", {}).get("pdp_features", [])
    fixas_validas = [a for a in fixas if a in X.columns and X[a].dtype.kind in "fi"]
    # União preservando ordem: top-3 primeiro, depois as fixas ainda não incluídas.
    alvos_pdp = top_numericas + [a for a in fixas_validas if a not in top_numericas]
    for atributo in alvos_pdp:
        pdp = dependencia_parcial(modelo, X, atributo)
        pdp.to_csv(dir_tab / f"pdp_{atributo}.csv", index=False)
    print(f"\nDependência parcial exportada para: {alvos_pdp}")

    shap_out = shap_valores(modelo, X, semente=semente)
    if shap_out is None:
        print("\nSHAP indisponível (pacote não instalado); etapa SHAP ignorada.")
    else:
        contrib, nomes = shap_out
        grafico_importancia(list(nomes), list(contrib),
                            caminho_figura("shap_resumo.pdf", config),
                            titulo=f"Importância média |SHAP| ({melhor})")
        print("\nResumo SHAP exportado.")

    # KM estratificado (log-rank) de apoio à discussão da desintermediação (Seção 5.4):
    # (a) fidelização entre clientes recorrentes; (b) bônus inicial (produtividade).
    p_km = km_estratificado_reputacao(df, caminho_figura("km_estratificado.pdf", config))
    print(f"\nKM estratificado exportado. log-rank: fidelização p={p_km['p_fidelizacao']:.4f} "
          f"(entre clientes recorrentes) | bônus p={p_km['p_bonus']:.2e}")


if __name__ == "__main__":
    main()
