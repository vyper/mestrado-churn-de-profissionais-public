"""Gerador de base sintética com três tabelas ilustrativas do domínio.

Gera três tabelas sintéticas (profissionais, diárias e incidentes) e simula um
processo de sobrevivência latente por profissional. Permite exercitar todo o
pipeline (cohort -> pré-processamento -> modelagem -> avaliação) sem acesso aos
dados proprietários, garantindo a reprodutibilidade pública prevista na Seção 4
da Entrega 3. As colunas produzidas aqui definem o formato consumido por
`cohort.py`.

Colunas geradas (uma observação por linha):

taskers   : tasker_id, updated_at, state, created_at, current_sign_in_at,
            gender, birthdate, city, state_uf, category, plan, indication,
            payable_by_cash, latitude, longitude, score, score_punctuality,
            score_friendliness, retention_rate, preferential_rate,
            completed_jobs, completed_preferential_jobs
jobs      : tasker_id, job_id, job_date, check_in_at, check_out_at, work_time,
            feedback_score, final_payout, latitude, longitude,
            preferential, performed_by_preferential, job_state
incidents : tasker_id, incident_id, type, incident_at, job_date, time_delta,
            apply_penalty, penalty_amount, penalty_percentage, final_payout,
            disable_tasker, pre_check_in_at, check_in_at
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIAS = ["bronze", "prata", "ouro"]
PLANOS = ["avulso", "mensal", "preferencial"]
INDICACOES = ["organico", "indicacao", "midia_paga", "parceria"]
TIPOS_INCIDENTE = ["no_show", "delayed", "cancel"]

# Centro aproximado da cidade de São Paulo (coordenadas fictícias de referência).
SP_LAT, SP_LON = -23.55, -46.63


def _amostrar_perfil(rng: np.random.Generator, n: int) -> dict:
    """Atributos de perfil constantes por profissional, com ausências realistas."""
    genero = rng.choice(["feminino", "masculino"], size=n, p=[0.85, 0.15]).astype(object)
    genero[rng.random(n) < 0.10] = None  # cadastro incompleto
    idade = rng.integers(18, 65, size=n).astype(float)
    nascimento = [
        None if rng.random() < 0.08 else (datetime(2026, 1, 1) - timedelta(days=int(a) * 365))
        for a in idade
    ]
    lat = SP_LAT + rng.normal(0, 0.12, n)
    lon = SP_LON + rng.normal(0, 0.12, n)
    sem_geo = rng.random(n) < 0.05  # endereços não geocodificáveis
    lat[sem_geo] = np.nan
    lon[sem_geo] = np.nan
    return {
        "gender": genero,
        "birthdate": nascimento,
        "category": rng.choice(CATEGORIAS, size=n, p=[0.5, 0.35, 0.15]),
        "plan": rng.choice(PLANOS, size=n, p=[0.5, 0.3, 0.2]),
        "indication": rng.choice(INDICACOES, size=n),
        "latitude": lat,
        "longitude": lon,
        "n_servicos_oferecidos": rng.integers(1, 6, size=n),
        "photo_identity_state": rng.choice(
            ["approved", "pending", None], size=n, p=[0.7, 0.2, 0.1]),
        "photo_identity_score": np.where(
            rng.random(n) < 0.75, rng.uniform(0.5, 1.0, n), np.nan),
        "photo_identity_checked_at": [None] * n,
    }


def gerar_base_sintetica(config: dict, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Gera as três tabelas sintéticas a partir dos parâmetros do config.

    Retorna um dicionário {"taskers", "jobs", "incidents"} de DataFrames.
    """
    par = config["sintetico"]
    n = int(par["n_profissionais"])
    inicio = pd.Timestamp(par["data_inicio"])
    corte = pd.Timestamp(par["data_corte"])
    perfil = _amostrar_perfil(rng, n)

    # Datas de cadastro (lead) distribuídas ao longo da janela de observação.
    dias_janela = (corte - inicio).days
    created = [inicio + timedelta(days=int(d)) for d in rng.integers(0, dias_janela, size=n)]
    ativou = rng.random(n) < par["taxa_ativacao"]

    linhas_taskers: list[dict] = []
    linhas_jobs: list[dict] = []
    linhas_incidents: list[dict] = []
    job_id = 0
    incident_id = 0

    for i in range(n):
        base = {
            "tasker_id": i,
            "created_at": created[i],
            "city": "São Paulo",
            "state_uf": "SP",
            "gender": perfil["gender"][i],
            "birthdate": perfil["birthdate"][i],
            "category": perfil["category"][i],
            "plan": perfil["plan"][i],
            "indication": perfil["indication"][i],
            "latitude": perfil["latitude"][i],
            "longitude": perfil["longitude"][i],
            "n_servicos_oferecidos": int(perfil["n_servicos_oferecidos"][i]),
            "photo_identity_state": perfil["photo_identity_state"][i],
            "photo_identity_score": perfil["photo_identity_score"][i],
            "photo_identity_checked_at": perfil["photo_identity_checked_at"][i],
        }

        def registro_estado(state, quando, **extra):
            """Cria uma linha historizada de taskers (snapshot de estado)."""
            r = dict(base)
            r.update({"state": state, "updated_at": quando})
            r.update(extra)
            linhas_taskers.append(r)

        # Marco inicial: lead.
        registro_estado("lead", created[i], current_sign_in_at=None,
                         score=np.nan, score_punctuality=np.nan, score_friendliness=np.nan,
                         retention_rate=np.nan, preferential_rate=np.nan,
                         completed_jobs=0, completed_preferential_jobs=0)

        if not ativou[i]:
            # Não ativados: param no funil (onboarding), sem diárias. Cohort os exclui.
            if rng.random() < 0.6:
                registro_estado("onboarding", created[i] + timedelta(days=int(rng.integers(1, 10))),
                                current_sign_in_at=None, score=np.nan, score_punctuality=np.nan,
                                score_friendliness=np.nan, retention_rate=np.nan,
                                preferential_rate=np.nan, completed_jobs=0,
                                completed_preferential_jobs=0)
            continue

        # --- Profissional ativado ---
        atraso_funil = int(rng.gamma(shape=2.0, scale=7.0)) + 1  # dias cadastro->ativação
        ativacao = created[i] + timedelta(days=atraso_funil)
        primeiro_login = ativacao - timedelta(days=int(rng.integers(0, 5)))
        registro_estado("onboarding", created[i] + timedelta(days=max(1, atraso_funil // 2)),
                        current_sign_in_at=primeiro_login, score=np.nan, score_punctuality=np.nan,
                        score_friendliness=np.nan, retention_rate=np.nan, preferential_rate=np.nan,
                        completed_jobs=0, completed_preferential_jobs=0)

        # Tempo latente até o churn (Weibull), a partir da ativação.
        offset_churn = float(par["churn_escala_weibull"] *
                             rng.weibull(par["churn_forma_weibull"]))
        churn_date = ativacao + timedelta(days=offset_churn)
        ativo_no_corte = churn_date >= corte
        fim_atividade = min(churn_date, corte)

        # Geração das diárias (recorrência ~ a cada 7 dias em média).
        datas_jobs: list[pd.Timestamp] = []
        t = ativacao + timedelta(days=float(rng.exponential(5)))
        while t < fim_atividade:
            datas_jobs.append(t)
            t = t + timedelta(days=float(rng.exponential(7)))

        n_pref = 0
        for jd in datas_jobs:
            job_id += 1
            work = float(np.clip(rng.normal(240, 50), 60, 600))  # minutos
            checkin = jd + timedelta(minutes=float(rng.normal(0, 20)))
            checkout = checkin + timedelta(minutes=work)
            nota = float(np.clip(rng.normal(4.5, 0.6), 1, 5))
            final_payout = float(np.clip(rng.normal(120, 30), 40, 400))
            pref = bool(rng.random() < 0.3)
            n_pref += int(pref)
            # Local da diária: perto de casa quando há geo; senão centro de SP.
            base_lat = base["latitude"] if not np.isnan(base["latitude"]) else SP_LAT
            base_lon = base["longitude"] if not np.isnan(base["longitude"]) else SP_LON
            servico = rng.choice(["cleaning", "express_cleaning", "heavy_cleaning",
                                   "furniture_assembly", "ironing"])
            tem_bonus = rng.random() < 0.15
            confirmou = rng.random() < 0.85
            linhas_jobs.append({
                "tasker_id": i, "job_id": job_id, "job_date": jd,
                "check_in_at": checkin, "check_out_at": checkout, "work_time": work,
                "feedback_score": nota, "final_payout": final_payout,
                "latitude": base_lat + rng.normal(0, 0.05),
                "longitude": base_lon + rng.normal(0, 0.05),
                "preferential": pref, "performed_by_preferential": pref,
                "job_state": "completed",
                "service": servico,
                "subscription": bool(rng.random() < 0.4),
                "bonus_payout": float(rng.uniform(10, 50)) if tem_bonus else 0.0,
                "confirmed_at": (jd - timedelta(days=int(rng.integers(1, 5))))
                                 if confirmou else None,
            })

            # Incidente esparso associado à diária.
            if rng.random() < par["prob_incidente_por_diaria"]:
                incident_id += 1
                tipo = rng.choice(TIPOS_INCIDENTE, p=[0.2, 0.6, 0.2])
                atraso_seg = float(abs(rng.normal(1800, 900))) if tipo == "delayed" else 0.0
                penaliza = bool(rng.random() < 0.5)
                linhas_incidents.append({
                    "tasker_id": i, "incident_id": incident_id, "type": tipo,
                    "incident_at": jd, "job_date": jd, "time_delta": atraso_seg,
                    "apply_penalty": penaliza,
                    "penalty_amount": float(rng.uniform(10, 50)) if penaliza else 0.0,
                    "penalty_percentage": float(rng.uniform(0.1, 0.5)) if penaliza else 0.0,
                    "final_payout": final_payout, "disable_tasker": False,
                    "pre_check_in_at": jd, "check_in_at": checkin,
                })

        n_jobs = len(datas_jobs)
        ultima_atividade = datas_jobs[-1] if datas_jobs else ativacao

        # Reputação consolidada (esparsa para quem tem poucas diárias).
        if n_jobs >= 3:
            score_p = float(np.clip(rng.normal(4.4, 0.4), 1, 5))
            score_f = float(np.clip(rng.normal(4.6, 0.3), 1, 5))
            retencao = float(np.clip(rng.normal(0.6, 0.2), 0, 1))
            pref_rate = n_pref / n_jobs
            # Scores por serviço: cada profissional avaliada em 1-3 serviços.
            servicos_aval = int(rng.integers(1, 4))
            sc_limpeza       = float(np.clip(rng.normal(4.5, 0.4), 1, 5)) if servicos_aval >= 1 else np.nan
            sc_express       = float(np.clip(rng.normal(4.4, 0.4), 1, 5)) if servicos_aval >= 2 else np.nan
            sc_pesada        = float(np.clip(rng.normal(4.3, 0.5), 1, 5)) if servicos_aval >= 3 else np.nan
            sc_montagem      = np.nan
            sc_passadoria    = np.nan
            n_limpeza        = int(rng.integers(3, 20)) if not np.isnan(sc_limpeza) else 0
            n_express        = int(rng.integers(1, 10)) if not np.isnan(sc_express) else 0
            n_pesada         = int(rng.integers(1, 5))  if not np.isnan(sc_pesada)  else 0
        else:
            score_p = score_f = retencao = np.nan
            pref_rate = np.nan
            sc_limpeza = sc_express = sc_pesada = sc_montagem = sc_passadoria = np.nan
            n_limpeza = n_express = n_pesada = 0

        agregados = dict(
            score_punctuality=score_p, score_friendliness=score_f,
            retention_rate=retencao, preferential_rate=pref_rate,
            completed_jobs=n_jobs, completed_preferential_jobs=n_pref,
            current_sign_in_at=ultima_atividade,
            score_limpeza=sc_limpeza,       n_aval_limpeza=n_limpeza,
            score_limpeza_express=sc_express, n_aval_limpeza_express=n_express,
            score_limpeza_pesada=sc_pesada,  n_aval_limpeza_pesada=n_pesada,
            score_montagem=sc_montagem,      n_aval_montagem=0,
            score_passadoria=sc_passadoria,  n_aval_passadoria=0,
        )

        # Marco de ativação (enabled) com os agregados consolidados.
        registro_estado("enabled", ativacao, **agregados)

        # Suspensão temporária: ~12% das profissionais ativas passam por um
        # período de suspended antes de retornar a enabled.
        dias_ativos = (fim_atividade - ativacao).days
        if dias_ativos > 14 and rng.random() < par.get("prob_suspensao", 0.12):
            offset_susp = int(rng.integers(7, max(8, dias_ativos // 2)))
            duracao_susp = int(rng.integers(3, 22))
            inicio_susp = ativacao + timedelta(days=offset_susp)
            retorno_susp = inicio_susp + timedelta(days=duracao_susp)
            if retorno_susp < fim_atividade:
                registro_estado("suspended", inicio_susp, **agregados)
                registro_estado("enabled", retorno_susp, **agregados)

        if not ativo_no_corte:
            involuntaria = rng.random() < par["prob_inativacao_involuntaria"]
            if involuntaria:
                # Inativação por justa causa: incidente crítico + disable_tasker.
                incident_id += 1
                linhas_incidents.append({
                    "tasker_id": i, "incident_id": incident_id, "type": "no_show",
                    "incident_at": churn_date, "job_date": ultima_atividade,
                    "time_delta": 0.0, "apply_penalty": True,
                    "penalty_amount": float(rng.uniform(50, 150)), "penalty_percentage": 1.0,
                    "final_payout": 0.0, "disable_tasker": True,
                    "pre_check_in_at": None, "check_in_at": None,
                })
                registro_estado("disabled", churn_date, **agregados)
            elif rng.random() < 0.5:
                # Churn voluntário com inativação administrativa (sem incidente crítico).
                lag = timedelta(days=float(rng.uniform(60, 120)))
                registro_estado("disabled", min(ultima_atividade + lag, corte), **agregados)
            # Caso restante: churn puro por inatividade (sem registro 'disabled').

    taskers = pd.DataFrame(linhas_taskers)
    jobs = pd.DataFrame(linhas_jobs)
    incidents = pd.DataFrame(linhas_incidents)
    # Tipos temporais coerentes.
    for col in ["updated_at", "created_at", "current_sign_in_at"]:
        taskers[col] = pd.to_datetime(taskers[col])
    for col in ["job_date", "check_in_at", "check_out_at"]:
        jobs[col] = pd.to_datetime(jobs[col])
    if not incidents.empty:
        for col in ["incident_at", "job_date", "pre_check_in_at", "check_in_at"]:
            incidents[col] = pd.to_datetime(incidents[col])
    return {"taskers": taskers, "jobs": jobs, "incidents": incidents}
