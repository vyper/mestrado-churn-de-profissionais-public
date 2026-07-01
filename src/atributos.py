"""Engenharia de atributos na janela de landmarking.

Materializa os grupos de covariáveis do mapeamento técnico (Entrega 1),
agregando diárias e incidentes ESTRITAMENTE dentro da janela inicial
[ativacao, landmark] de cada profissional. Essa restrição é a prevenção de
vazamento temporal descrita na Seção 2 da Entrega 3: nenhum preditor usa
informação posterior ao instante de predição (o landmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAIO_TERRA_KM = 6371.0

_SCORES_SERVICO = [
    ("score_limpeza",        "n_aval_limpeza"),
    ("score_limpeza_express","n_aval_limpeza_express"),
    ("score_limpeza_pesada", "n_aval_limpeza_pesada"),
    ("score_montagem",       "n_aval_montagem"),
    ("score_passadoria",     "n_aval_passadoria"),
]


def _haversine(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos (vetorizada)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def _idade(nascimento: pd.Series, referencia: pd.Timestamp) -> pd.Series:
    return ((referencia - pd.to_datetime(nascimento)).dt.days / 365.25).round(1)


def atributos_perfil_e_funil(cohort: pd.DataFrame) -> pd.DataFrame:
    """Grupos de perfil/demografia, ciclo de vida/funil e onboarding."""
    ref = pd.Timestamp("2026-01-01")
    out = pd.DataFrame({"tasker_id": cohort["tasker_id"]})
    out["idade"] = _idade(cohort["birthdate"], ref)
    out["genero"] = cohort["gender"]
    out["plano"] = cohort["plan"]
    out["canal_indicacao"] = cohort["indication"]
    # Decomposição do funil em dois intervalos independentes:
    #   dias_cad_onb  — motivação inicial: quanto o candidato demorou para COMEÇAR o processo.
    #   dias_onb_atv  — velocidade de conclusão: tempo para passar de onboarding a ativado.
    # Ambos são anteriores ao landmark (datas pré-ativação), sem risco de leakage.
    # dias_cadastro_ativacao é mantido como soma dos dois para compatibilidade/referência.
    if "onboarding_at" in cohort.columns:
        out["dias_cad_onb"] = (
            (cohort["onboarding_at"] - cohort["criacao_at"]).dt.days.clip(lower=0))
        out["dias_onb_atv"] = (
            (cohort["ativacao_at"] - cohort["onboarding_at"]).dt.days.clip(lower=0))
    out["dias_cadastro_ativacao"] = (
        (cohort["ativacao_at"] - cohort["criacao_at"]).dt.days.clip(lower=0))
    # Diversificação de serviços oferecidos.
    if "n_servicos_oferecidos" in cohort.columns:
        out["n_servicos_oferecidos"] = cohort["n_servicos_oferecidos"]
    # Verificação de identidade (onboarding): binário + velocidade de conclusão.
    # Anti-leakage: só usa a informação se a verificação ocorreu ANTES do landmark.
    # photo_identity_state e photo_identity_checked_at vêm do registro mais recente
    # do tasker (derivar_marcos usa .last()), portanto podem estar no futuro do landmark.
    tem_landmark = "landmark_at" in cohort.columns
    if "photo_identity_checked_at" in cohort.columns and "ativacao_at" in cohort.columns:
        checked = pd.to_datetime(cohort["photo_identity_checked_at"])
        if tem_landmark:
            # Mascara verificações futuras ao landmark — informação não disponível
            # no instante de predição.
            antes_landmark = checked <= cohort["landmark_at"]
            checked = checked.where(antes_landmark)
        delta = (checked - cohort["ativacao_at"]).dt.days
        out["dias_ate_verificacao_identidade"] = delta.clip(lower=0)
    if "photo_identity_state" in cohort.columns:
        if tem_landmark and "photo_identity_checked_at" in cohort.columns:
            # Só considera identidade verificada se a data de checagem é <= landmark.
            checked_raw = pd.to_datetime(cohort["photo_identity_checked_at"])
            antes = checked_raw <= cohort["landmark_at"]
            aprovada = cohort["photo_identity_state"].eq("approved")
            out["identidade_verificada"] = (aprovada & antes).astype(float)
        else:
            out["identidade_verificada"] = (
                cohort["photo_identity_state"].eq("approved").astype(float))
    return out


def atributos_janela(cohort: pd.DataFrame, jobs: pd.DataFrame,
                     incidents: pd.DataFrame) -> pd.DataFrame:
    """Agrega produtividade, reputação, deslocamento e conduta na janela de landmark.

    Para cada profissional, considera apenas diárias e incidentes com data em
    [ativacao_at, landmark_at]. Incidentes ausentes significam contagem zero.
    """
    janela = cohort[["tasker_id", "ativacao_at", "landmark_at", "latitude", "longitude"]].copy()

    # --- Diárias na janela ---
    j = jobs.merge(janela[["tasker_id", "ativacao_at", "landmark_at",
                           "latitude", "longitude"]], on="tasker_id", how="inner")
    na_janela = (j["job_date"] >= j["ativacao_at"]) & (j["job_date"] <= j["landmark_at"])
    j = j[na_janela].copy()
    # Distância casa <-> diária (deslocamento).
    j["distancia_km"] = _haversine(j["latitude_y"], j["longitude_y"],
                                   j["latitude_x"], j["longitude_x"])
    # Antecedência da confirmação: dias entre confirmed_at e a data da diária.
    if "confirmed_at" in j.columns and "job_date" in j.columns:
        j["antecedencia_confirmacao_dias"] = (
            (j["job_date"] - pd.to_datetime(j["confirmed_at"])).dt.days.clip(lower=0))

    # nota_media_diarias: usa apenas avaliações genuínas.
    # Jobs performed_by_preferential recebem score 5 automático quando o cliente
    # não avalia ativamente — incluí-los infla a nota para profissionais com mais
    # clientes preferenciais, criando confundimento com tempo de plataforma.
    # feedback_score == 0 significa "não avaliado" (ausência de rating, não nota zero).
    # Filtramos performed_by_preferential=False e score > 0 para obter avaliações reais.
    if "performed_by_preferential" in j.columns:
        j_genuino = j[(j["performed_by_preferential"] == False) &   # noqa: E712
                      (j["feedback_score"] > 0)]
    else:
        j_genuino = j[j["feedback_score"] > 0]

    nota_genuina = (j_genuino.groupby("tasker_id")["feedback_score"]
                              .agg(nota_media_diarias="mean",
                                   n_diarias_avaliadas="count"))

    agg_dict = dict(
        n_diarias=("job_id", "count"),
        n_diarias_pref=("preferential", "sum"),
        tempo_trabalhado_total=("work_time", "sum"),
        tempo_trabalhado_medio=("work_time", "mean"),
        remuneracao_total=("final_payout", "sum"),
        remuneracao_media=("final_payout", "mean"),
        distancia_media_km=("distancia_km", "mean"),
        distancia_mediana_km=("distancia_km", "median"),
        distancia_max_km=("distancia_km", "max"),
        distancia_p90_km=("distancia_km", lambda s: s.quantile(0.9)),
        distancia_desvio_km=("distancia_km", "std"),
    )
    if "subscription" in j.columns:
        agg_dict["fracao_assinatura"] = ("subscription", "mean")
    if "service" in j.columns:
        agg_dict["n_tipos_servico"] = ("service", "nunique")
    if "bonus_payout" in j.columns:
        agg_dict["bonus_total"] = ("bonus_payout", "sum")
    if "antecedencia_confirmacao_dias" in j.columns:
        agg_dict["antecedencia_media_confirmacao_dias"] = (
            "antecedencia_confirmacao_dias", "mean")

    prod = j.groupby("tasker_id").agg(**agg_dict).join(nota_genuina)

    # --- Incidentes na janela (conduta/desengajamento) ---
    if not incidents.empty:
        inc = incidents.merge(janela[["tasker_id", "ativacao_at", "landmark_at"]],
                              on="tasker_id", how="inner")
        na_janela_inc = (inc["incident_at"] >= inc["ativacao_at"]) & \
                        (inc["incident_at"] <= inc["landmark_at"])
        # Exclui incidentes críticos (disable_tasker) do conjunto de preditores:
        # são usados apenas para censura, jamais como covariável (anti-circularidade).
        inc = inc[na_janela_inc & (inc["disable_tasker"] != True)].copy()  # noqa: E712
        inc["atraso_min"] = inc["time_delta"] / 60.0  # segundos -> minutos
        cond = inc.groupby("tasker_id").agg(
            n_incidentes=("incident_id", "count"),
            n_no_show=("type", lambda s: (s == "no_show").sum()),
            n_atraso=("type", lambda s: (s == "delayed").sum()),
            n_cancelamento=("type", lambda s: (s == "cancel").sum()),
            atraso_medio_min=("atraso_min", "mean"),
            atraso_max_min=("atraso_min", "max"),
            penalidade_total=("penalty_amount", "sum"),
        )
    else:
        cond = pd.DataFrame()

    base = janela[["tasker_id"]].set_index("tasker_id")
    feats = base.join(prod).join(cond)

    # Incidentes esparsos: ausência = contagem zero, não nulo (Seção 2 da Entrega 3).
    cols_zero = ["n_incidentes", "n_no_show", "n_atraso", "n_cancelamento",
                 "atraso_medio_min", "atraso_max_min", "penalidade_total",
                 "n_diarias", "n_diarias_pref", "bonus_total", "n_diarias_avaliadas"]
    for c in cols_zero:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
    # Taxas de conduta por diária.
    denom = feats["n_diarias"].replace(0, np.nan)
    feats["taxa_no_show"] = (feats["n_no_show"] / denom).fillna(0.0)
    feats["taxa_atraso"] = (feats["n_atraso"] / denom).fillna(0.0)
    feats["taxa_cancelamento"] = (feats["n_cancelamento"] / denom).fillna(0.0)
    feats["fracao_preferencial"] = (feats["n_diarias_pref"] / denom).fillna(0.0)
    return feats.reset_index()


def atributos_reputacao(cohort: pd.DataFrame) -> pd.DataFrame:
    """Reputação consolidada AS-OF-LANDMARK: dimensões transversais e scores por serviço.

    Os scores chegam já capturados NO landmark (cohort.snapshot_reputacao), sem
    vazamento temporal. Nulos são informativos (recém-ativadas sem avaliações
    consolidadas no landmark). Scores por serviço são agregados em
    nota_media_servicos (média ponderada pelo nº de avaliações), n_servicos_avaliados
    e nota_max_servico.
    """
    cols_base = ["score_punctuality", "score_friendliness",
                 "retention_rate", "preferential_rate"]
    out = cohort[["tasker_id"] + cols_base].copy()
    out = out.rename(columns={
        "score_punctuality": "pontualidade",
        "score_friendliness": "simpatia",
        "retention_rate": "taxa_retencao",
        "preferential_rate": "taxa_preferencial",
    })

    # Agrega scores por serviço: média ponderada, contagem e máximo.
    scores_cols = [s for s, _ in _SCORES_SERVICO if s in cohort.columns]
    totais_cols = [n for _, n in _SCORES_SERVICO if n in cohort.columns]
    if scores_cols:
        # to_numpy com na_value=np.nan converte pd.NA (BigQuery nullable) corretamente.
        scores = cohort[scores_cols].to_numpy(dtype=float, na_value=np.nan)
        totais = cohort[totais_cols].to_numpy(dtype=float, na_value=np.nan)
        # Zera pesos onde score é nulo.
        pesos = np.where(np.isnan(scores), 0.0, totais)
        soma_pesos = pesos.sum(axis=1)
        soma_pond = np.nansum(scores * pesos, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            media = np.where(soma_pesos > 0, soma_pond / soma_pesos, np.nan)
            maximo = np.where(
                (totais > 0).any(axis=1),
                np.nanmax(np.where(np.isnan(scores), -np.inf, scores), axis=1),
                np.nan,
            )
        out["nota_media_servicos"] = media
        out["n_servicos_avaliados"] = (totais > 0).sum(axis=1).astype(float)
        out["nota_max_servico"] = maximo

    return out


def construir_atributos(cohort: pd.DataFrame, jobs: pd.DataFrame,
                        incidents: pd.DataFrame) -> pd.DataFrame:
    """Monta a matriz de atributos por profissional (uma linha por tasker_id).

    Saída: cohort (tempo, evento) + todos os grupos de covariáveis, pronta para
    o pré-processamento. Mantém as colunas-alvo `tempo` e `evento`.
    """
    perfil = atributos_perfil_e_funil(cohort)
    janela = atributos_janela(cohort, jobs, incidents)
    reput = atributos_reputacao(cohort)

    df = cohort[["tasker_id", "tempo", "evento"]].copy()
    for parte in (perfil, janela, reput):
        df = df.merge(parte, on="tasker_id", how="left")
    return df
