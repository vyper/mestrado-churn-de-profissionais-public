"""Geração de figuras com formatação adequada para publicação científica.

Centraliza os gráficos exportados em resultados/figuras: curva de
Kaplan-Meier da cohort, diagrama de diferença crítica (Nemenyi) e gráficos de
importância de atributos. Usa matplotlib com backend não interativo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402


def curva_kaplan_meier(tempo, evento, destino: Path, titulo: str = "Curva de sobrevivência") -> Path:
    """Plota a curva de Kaplan-Meier global da cohort (probabilidade de permanência)."""
    t, s, conf = kaplan_meier_estimator(np.asarray(evento, dtype=bool),
                                        np.asarray(tempo, dtype=float),
                                        conf_type="log-log")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(t, s, where="post", color="#1f4e79", label="Kaplan-Meier")
    ax.fill_between(t, conf[0], conf[1], alpha=0.2, step="post", color="#1f4e79")
    ax.set_xlabel("Tempo desde o landmark (dias)")
    ax.set_ylabel("Probabilidade de permanência S(t)")
    ax.set_ylim(0, 1)
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    return destino


def diagrama_diferenca_critica(ranks: pd.Series, cd: float, destino: Path) -> Path:
    """Diagrama de diferença crítica (Demšar): eixo de ranks médios com barras
    conectando grupos de modelos SEM diferença significativa (distância <= CD).

    Sem título nem moldura (a legenda do artigo cobre o título), com margens
    generosas e vírgula decimal (pt-BR), para leitura limpa em publicação.
    """
    ranks = ranks.sort_values()
    modelos = list(ranks.index)
    valores = ranks.to_numpy().astype(float)
    lo, hi = float(valores.min()), float(valores.max())
    pad = max(0.35, cd * 0.6)
    x0, x1 = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.85, 1.35)
    ax.axis("off")

    # Eixo de ranks (número-linha) em y=0, com ticks e números abaixo.
    ax.hlines(0, x0, x1, color="black", lw=1.3, zorder=2)
    t0 = np.ceil(x0 / 0.2) * 0.2
    for t in np.round(np.arange(t0, x1 + 1e-9, 0.2), 2):
        ax.vlines(t, -0.05, 0.0, color="black", lw=1, zorder=2)
        ax.text(t, -0.14, f"{t:.1f}".replace(".", ","), ha="center", va="top", fontsize=9)
    ax.text((x0 + x1) / 2, -0.52, "Rank médio (1 = melhor)",
            ha="center", va="top", fontsize=10.5)

    # Modelos: ponto no eixo, conector fino e rótulo (nome + rank) acima.
    for m, v in zip(modelos, valores):
        ax.vlines(v, 0.0, 0.40, color="#1f4e79", lw=1.1, zorder=1)
        ax.scatter([v], [0.0], color="#1f4e79", s=48, zorder=3)
        ax.text(v, 0.46, f"{m}\n(rank {v:.2f})".replace(".", ","),
                ha="center", va="bottom", fontsize=9.5)

    # Barras de clique: grupos maximais cuja amplitude de rank <= CD (não diferem).
    grupos, i = [], 0
    while i < len(valores):
        j = i
        while j + 1 < len(valores) and valores[j + 1] - valores[i] <= cd:
            j += 1
        grupos.append((valores[i], valores[max(j, i)]))
        i = max(j, i) + 1
    yb = 0.16
    for a, b in grupos:
        if b > a:  # só desenha se conecta >= 2 modelos
            ax.plot([a, b], [yb, yb], color="crimson", lw=5,
                    solid_capstyle="round", zorder=4)
            yb += 0.12

    # Referência da distância crítica (topo), afastada dos rótulos.
    xr = x0 + 0.03 * (x1 - x0)
    yr = 1.18
    ax.plot([xr, xr + cd], [yr, yr], color="black", lw=2, zorder=3)
    ax.vlines([xr, xr + cd], yr - 0.04, yr + 0.04, color="black", lw=1.3, zorder=3)
    ax.text(xr + cd / 2, yr + 0.06, f"CD = {cd:.2f}".replace(".", ","),
            ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    return destino


def km_estratificado_reputacao(df: pd.DataFrame, destino: Path,
                               col_fidelizacao: str = "taxa_preferencial",
                               col_bonus: str = "bonus_total") -> dict:
    """KM estratificado de apoio à discussão da desintermediação (Seção 5.4).

    Painel (a): fidelização de clientes ENTRE profissionais com clientes recorrentes
    (`col_fidelizacao` > 0), dividida na mediana. É o teste conceitualmente correto
    da desintermediação — só pode migrar "para fora" quem tem relação recorrente
    estabelecida —, e excluir os casos de fidelização ausente/zero evita o artefato
    do balde de dados ausentes (churners precoces com pouco histórico). Painel (b):
    bônus inicial (produtividade) em tercis. Cada painel traz o p-valor de log-rank.
    Retorna {'p_fidelizacao': float, 'p_bonus': float}.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    kmf = KaplanMeierFitter()

    # (a) Fidelização apenas entre quem tem clientes recorrentes (col > 0).
    tp = df[col_fidelizacao]
    pos = (tp.notna()) & (tp > 0)
    med = tp[pos].median()
    grp = pd.Series(index=df.index[pos], dtype=object)
    grp[tp[pos] <= med] = "Fidelização baixa"
    grp[tp[pos] > med] = "Fidelização alta"
    for nome, cor in [("Fidelização baixa", "#1f77b4"), ("Fidelização alta", "#ff7f0e")]:
        idx = grp.index[grp == nome]
        kmf.fit(df.loc[idx, "tempo"], df.loc[idx, "evento"], label=f"{nome} (n={len(idx)})")
        kmf.plot_survival_function(ax=axes[0], ci_show=True, color=cor)
    p_fid = float(multivariate_logrank_test(
        df.loc[pos, "tempo"], grp, df.loc[pos, "evento"]).p_value)
    ns = " (n.s.)" if p_fid >= 0.05 else ""
    axes[0].set_title(f"(a) Fidelização | clientes recorrentes — log-rank p={p_fid:.2f}{ns}")

    # (b) Bônus inicial em tercis.
    b = df[col_bonus]
    q = b.quantile([1 / 3, 2 / 3]).values
    gb = pd.cut(b, [-np.inf, q[0], q[1], np.inf], labels=["Baixo", "Médio", "Alto"])
    for nome, cor in [("Baixo", "#1f77b4"), ("Médio", "#ff7f0e"), ("Alto", "#2ca02c")]:
        idx = df.index[gb == nome]
        kmf.fit(df.loc[idx, "tempo"], df.loc[idx, "evento"], label=f"{nome} (n={len(idx)})")
        kmf.plot_survival_function(ax=axes[1], ci_show=False, color=cor)
    p_bon = float(multivariate_logrank_test(df["tempo"], gb, df["evento"]).p_value)
    p_txt = "< 0,001" if p_bon < 0.001 else f"{p_bon:.3f}".replace(".", ",")
    axes[1].set_title(f"(b) Bônus inicial (produtividade) — log-rank p {p_txt}")

    for ax in axes:
        ax.set_xlabel("Tempo desde o landmark (dias)")
        ax.set_ylabel("S(t)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    return {"p_fidelizacao": p_fid, "p_bonus": p_bon}


def grafico_importancia(nomes, valores, destino: Path, titulo: str, n: int = 15) -> Path:
    """Gráfico de barras horizontais dos n atributos mais importantes."""
    ordem = np.argsort(valores)[-n:]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(ordem))))
    ax.barh([nomes[i] for i in ordem], [valores[i] for i in ordem], color="#1f4e79")
    ax.set_xlabel("Importância")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    return destino
