"""Construção da cohort e definição operacional do alvo de sobrevivência.

Reproduz a Seção 1 da Entrega 3: deriva os marcos do ciclo de vida, aplica o
landmarking (origem do tempo = ativação + 30 dias), define o evento (churn
voluntário por inatividade de W dias ou inativação não-crítica) e a censura
(profissionais ainda ativos e inativações involuntárias), e trata o viés de
borda por uma janela de expurgo ao final da observação.

Nota de operacionalização da janela de expurgo: para que o critério de
inatividade de W dias possa ser avaliado de forma justa, a observação é
encerrada em `corte - W` (corte efetivo). Profissionais ainda ativos nessa
janela final são censurados à direita no corte efetivo, em vez de descartados,
preservando a população censurada; o efeito prático sobre o viés de borda é o
mesmo descrito na Entrega 3 (nenhum profissional é rotulado como churn sem W
dias de inatividade observável).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def _normalizar(texto: str) -> str:
    """Remove acentos e baixa caixa (espelha o REGEXP_REPLACE/NORMALIZE da extração)."""
    if not isinstance(texto, str):
        return ""
    nfd = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").strip()


def derivar_marcos(taskers: pd.DataFrame) -> pd.DataFrame:
    """Deriva, por profissional, os marcos do ciclo de vida via MIN(updated_at) por estado.

    Retorna uma linha por tasker_id com as datas dos marcos e os atributos de
    perfil (constantes), além dos agregados do último registro historizado.
    """
    g = taskers.sort_values("updated_at")

    def primeiro_em(estado: str) -> pd.Series:
        sub = g[g["state"] == estado].groupby("tasker_id")["updated_at"].min()
        return sub

    marcos = pd.DataFrame({"tasker_id": taskers["tasker_id"].unique()}).set_index("tasker_id")
    marcos["criacao_at"] = g.groupby("tasker_id")["created_at"].min()
    marcos["onboarding_at"] = primeiro_em("onboarding")
    marcos["ativacao_at"] = primeiro_em("enabled")
    marcos["suspensao_at"] = primeiro_em("suspended")
    marcos["inativacao_at"] = primeiro_em("disabled")
    # Primeiro login: primeiro current_sign_in_at preenchido.
    login = (g.dropna(subset=["current_sign_in_at"])
               .groupby("tasker_id")["current_sign_in_at"].min())
    marcos["primeiro_login_at"] = login

    # Atributos de perfil (aproximadamente constantes), do registro mais recente.
    perfil_cols = ["created_at", "city", "state_uf", "gender", "birthdate", "category",
                   "plan", "indication", "latitude", "longitude",
                   "n_servicos_oferecidos",
                   "photo_identity_state", "photo_identity_score", "photo_identity_checked_at"]
    # ATENÇÃO (anti-vazamento): as colunas de reputação/score NÃO são extraídas
    # aqui. Elas evoluem no tempo (acúmulo de avaliações) e, se tomadas do registro
    # mais recente (.last()), incorporariam informação POSTERIOR ao landmark — como
    # a reputação evolui no tempo, o valor mais recente quase nunca coincide com o
    # do landmark. São capturadas AS-OF-LANDMARK em `snapshot_reputacao`.
    ultimo = g.groupby("tasker_id").last()
    for c in perfil_cols:
        if c in ultimo.columns:
            marcos[c] = ultimo[c]
    return marcos.reset_index()


def filtrar_cohort(marcos: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Restringe à capital de SP (por residência) e a profissionais que ativaram."""
    par = config["cohort"]
    cidade_alvo = _normalizar(par["cidade"])
    cidade_ok = marcos["city"].map(_normalizar) == cidade_alvo
    estado_ok = marcos["state_uf"].astype(str).str.upper() == par["estado"].upper()
    ativou = marcos["ativacao_at"].notna()
    return marcos[cidade_ok & estado_ok & ativou].copy()


def _ultima_atividade(jobs: pd.DataFrame) -> pd.Series:
    """Data da última diária concluída por profissional."""
    return jobs.groupby("tasker_id")["job_date"].max()


def _involuntarios(incidents: pd.DataFrame) -> set[int]:
    """Profissionais com inativação involuntária (incidente crítico disable_tasker=True)."""
    if incidents.empty:
        return set()
    crit = incidents[incidents["disable_tasker"] == True]  # noqa: E712
    return set(crit["tasker_id"].unique())


def definir_alvo(marcos: pd.DataFrame, jobs: pd.DataFrame, incidents: pd.DataFrame,
                 config: dict) -> pd.DataFrame:
    """Aplica landmarking, define (T, evento) e as regras de censura e exclusão.

    Retorna a cohort final: uma linha por profissional com `tempo` (dias desde o
    landmark), `evento` (bool), além das datas auxiliares para a agregação de
    atributos na janela de landmarking.
    """
    par = config["cohort"]
    landmark_dias = int(par["landmark_dias"])
    W = int(par["janela_inatividade_dias"])
    expurgo = int(par["janela_expurgo_dias"])
    corte = pd.Timestamp(par["data_corte"])
    corte_efetivo = corte - pd.Timedelta(days=expurgo)

    ultima_ativ = _ultima_atividade(jobs)
    involuntarios = _involuntarios(incidents)

    registros = []
    for _, prof in marcos.iterrows():
        tid = prof["tasker_id"]
        ativacao = prof["ativacao_at"]
        landmark = ativacao + pd.Timedelta(days=landmark_dias)
        ult = ultima_ativ.get(tid, pd.NaT)
        inativacao = prof["inativacao_at"]

        if tid in involuntarios:
            # Censura por inativação involuntária (justa causa).
            evento = False
            data_desfecho = min(inativacao, corte_efetivo) if pd.notna(inativacao) else corte_efetivo
        elif pd.isna(ult):
            # Ativado sem nenhuma diária: descartado (não chega a produzir).
            continue
        elif (corte_efetivo - ult).days >= W or (
                pd.notna(inativacao) and inativacao <= corte_efetivo):
            # Churn voluntário: inatividade >= W dias ou inativação administrativa.
            evento = True
            data_desfecho = ult
        else:
            # Ainda ativo na janela observável: censurado à direita no corte efetivo.
            evento = False
            data_desfecho = corte_efetivo

        # Exclusão: não atingiu o landmark (churn/censura antes da janela de observação).
        if data_desfecho < landmark:
            continue

        tempo = (data_desfecho - landmark).days
        if tempo < 0:
            continue

        registros.append({
            "tasker_id": tid,
            "tempo": float(tempo),
            "evento": bool(evento),
            "ativacao_at": ativacao,
            "landmark_at": landmark,
            "data_desfecho": data_desfecho,
        })

    cohort = pd.DataFrame(registros)
    return cohort


# Colunas de reputação consolidada que evoluem no tempo e, por isso, precisam ser
# capturadas NO landmark (e não no registro mais recente) para evitar vazamento.
_REPUT_COLS = [
    "score_punctuality", "score_friendliness", "retention_rate", "preferential_rate",
    "score_limpeza", "n_aval_limpeza",
    "score_limpeza_express", "n_aval_limpeza_express",
    "score_limpeza_pesada", "n_aval_limpeza_pesada",
    "score_montagem", "n_aval_montagem",
    "score_passadoria", "n_aval_passadoria",
]


def snapshot_reputacao(taskers: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """Reputação consolidada AS-OF-LANDMARK: uma linha por profissional.

    Para cada profissional, toma o ÚLTIMO registro historizado de `taskers` com
    `updated_at <= landmark_at`. Assim, os scores refletem o estado conhecido NO
    landmark (dia 30 após a ativação), e não o registro mais recente (posterior ao
    landmark), eliminando o vazamento temporal na covariável dominante do modelo.
    `landmarks` deve conter as colunas (tasker_id, landmark_at).
    """
    cols = [c for c in _REPUT_COLS if c in taskers.columns]
    t = taskers[["tasker_id", "updated_at"] + cols].copy()
    t["updated_at"] = pd.to_datetime(t["updated_at"])
    tj = t.merge(landmarks[["tasker_id", "landmark_at"]], on="tasker_id", how="inner")
    tj = tj[tj["updated_at"] <= pd.to_datetime(tj["landmark_at"])]
    snap = tj.sort_values("updated_at").groupby("tasker_id")[cols].last()
    return snap.reset_index()


def montar_cohort(taskers: pd.DataFrame, jobs: pd.DataFrame, incidents: pd.DataFrame,
                  config: dict) -> pd.DataFrame:
    """Pipeline completo: marcos -> filtro de cohort -> alvo com landmarking.

    Retorna a cohort com (tempo, evento), os marcos de perfil e a reputação
    consolidada capturada AS-OF-LANDMARK (anti-vazamento), prontos para a
    engenharia de atributos.
    """
    marcos = derivar_marcos(taskers)
    cohort_marcos = filtrar_cohort(marcos, config)
    alvo = definir_alvo(cohort_marcos, jobs, incidents, config)
    # Junta os atributos de perfil/marcos à cohort com alvo definido.
    cohort = alvo.merge(cohort_marcos.drop(columns=["ativacao_at"]), on="tasker_id", how="left")
    # Reputação consolidada NO landmark (não no registro mais recente): sem vazamento.
    reput = snapshot_reputacao(taskers, alvo[["tasker_id", "landmark_at"]])
    cohort = cohort.merge(reput, on="tasker_id", how="left")
    return cohort


def resumo_cohort(cohort: pd.DataFrame) -> dict:
    """Estatísticas-resumo da cohort (tamanho, taxa de evento/censura, tempo)."""
    n = len(cohort)
    n_evt = int(cohort["evento"].sum())
    return {
        "n_profissionais": n,
        "n_eventos": n_evt,
        "n_censurados": n - n_evt,
        "taxa_evento": n_evt / n if n else float("nan"),
        "tempo_mediano": float(cohort["tempo"].median()) if n else float("nan"),
        "tempo_max": float(cohort["tempo"].max()) if n else float("nan"),
    }
