"""Experimento 03: treina e avalia os três modelos sob o protocolo da Entrega 3.

Validação cruzada k-fold estratificada pelo indicador de evento, com busca
aninhada de hiperparâmetros dentro de cada treino (sem vazamento). Reporta,
por partição, C-index de Harrell e de Uno, AUC(t) e IBS. Salva:
  - resultados/tabelas/metricas_por_fold.csv  (insumo da etapa estatística)
  - resultados/tabelas/comparacao_modelos.(tex|csv)
  - resultados/tabelas/hiperparametros.(tex|csv)
  - resultados/modelos/<modelo>.joblib         (modelos finais p/ interpretabilidade)

Uso:
    python experimentos/03_treinar_avaliar.py [--rapido]
"""
from __future__ import annotations

import os

# Capa as threads BLAS/OpenMP/Accelerate por worker ANTES de importar numpy/sklearn,
# para que a paralelização ocorra no nível dos candidatos da busca (config:
# paralelismo.n_jobs) sem oversubscription — mantendo a máquina responsiva.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.avaliacao import avaliar, tempos_avaliacao
from src.modelos import construir_alvo_estruturado, fabricas_modelos
from src.preprocessamento import construir_preprocessador
from src.utils import (carregar_config, carregar_parquet, caminho_abs,
                       exportar_tabela_latex, fixar_sementes, n_jobs,
                       registrar_execucao)

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _busca(pipe, espaco, X, y, n_iter, semente, nucleos):
    """Busca aleatória de hiperparâmetros (CV interna), pontuada pelo C de Harrell."""
    inner = KFold(n_splits=3, shuffle=True, random_state=semente)
    busca = RandomizedSearchCV(pipe, espaco, n_iter=n_iter, cv=inner,
                               random_state=semente, n_jobs=nucleos, refit=True,
                               error_score="raise")
    busca.fit(X, y)
    return busca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="reduz folds/iterações para validação rápida do pipeline")
    args = ap.parse_args()

    config = carregar_config()
    semente = config["semente"]
    fixar_sementes(semente)

    n_folds = 3 if args.rapido else config["protocolo"]["n_folds"]
    n_iter = 5 if args.rapido else config["protocolo"]["n_iter_busca"]
    quantis = config["protocolo"]["tempos_avaliacao_quantis"]
    nucleos = n_jobs(config)
    print(f"Paralelismo: {nucleos} núcleo(s) na busca (BLAS por worker capado a 1).",
          flush=True)

    df = carregar_parquet(f"{config['caminhos']['dados_processado']}/cohort.parquet")
    X = df.drop(columns=["tempo", "evento"])
    y = construir_alvo_estruturado(df["tempo"], df["evento"])
    evento = df["evento"].to_numpy()

    fabricas = fabricas_modelos(config, semente)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semente)

    registros = []
    for nome, (estimador, espaco, escalar) in fabricas.items():
        print(f"[{nome}] avaliando {n_folds} folds (busca de {n_iter} iterações)...",
              flush=True)
        for k, (idx_tr, idx_te) in enumerate(skf.split(X, evento)):
            X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
            y_tr, y_te = y[idx_tr], y[idx_te]
            pre = construir_preprocessador(X_tr, escalar=escalar)
            pipe = Pipeline([("pre", pre), ("modelo", estimador)])
            busca = _busca(pipe, espaco, X_tr, y_tr, n_iter, semente, nucleos)
            tempos, tau = tempos_avaliacao(y_tr, y_te, quantis)
            metricas = avaliar(busca.best_estimator_, X_te, y_tr, y_te, tempos, tau)
            metricas.update({"modelo": nome, "fold": k})
            registros.append(metricas)
            print(f"  {nome:18s} fold {k + 1}/{n_folds}: C-Uno={metricas['c_uno']:.3f} "
                  f"AUC={metricas['auc_media']:.3f} IBS={metricas['ibs']:.3f}", flush=True)

    por_fold = pd.DataFrame(registros)
    dir_tab = caminho_abs(config["caminhos"]["resultados_tabelas"])
    dir_tab.mkdir(parents=True, exist_ok=True)
    por_fold.to_csv(dir_tab / "metricas_por_fold.csv", index=False)

    # Tabela-resumo: média ± desvio por modelo.
    metricas_cols = ["c_harrell", "c_uno", "auc_media", "ibs"]
    resumo = (por_fold.groupby("modelo")[metricas_cols]
              .agg(["mean", "std"]).round(4))
    resumo.columns = [f"{m}_{s}" for m, s in resumo.columns]
    comparacao = resumo.reset_index()
    exportar_tabela_latex(comparacao, "comparacao_modelos", config,
                          caption="Comparação dos modelos (média e desvio sobre as partições).",
                          label="tab:comparacao")

    # Ajuste final em toda a cohort + hiperparâmetros vencedores e modelos salvos.
    dir_mod = caminho_abs("resultados/modelos")
    dir_mod.mkdir(parents=True, exist_ok=True)
    linhas_hp = []
    print("Ajuste final na cohort completa (salvando modelos)...", flush=True)
    for nome, (estimador, espaco, escalar) in fabricas.items():
        pre = construir_preprocessador(X, escalar=escalar)
        pipe = Pipeline([("pre", pre), ("modelo", estimador)])
        busca = _busca(pipe, espaco, X, y, n_iter, semente, nucleos)
        joblib.dump(busca.best_estimator_, dir_mod / f"{nome.replace(' ', '_')}.joblib")
        melhores = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
        linhas_hp.append({"modelo": nome, "hiperparametros": str(melhores)})
    exportar_tabela_latex(pd.DataFrame(linhas_hp), "hiperparametros", config,
                          caption="Hiperparâmetros vencedores por modelo.",
                          label="tab:hiperparametros")

    registrar_execucao("treino_avaliacao", config,
                       metricas={"resumo": comparacao.to_dict(orient="records")})
    print("\nResumo (médias):")
    print(comparacao.to_string(index=False))


if __name__ == "__main__":
    main()
