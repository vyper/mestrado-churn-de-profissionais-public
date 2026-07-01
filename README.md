# Análise de sobrevivência aplicada ao churn de profissionais (diaristas)

Código dos experimentos do artigo da disciplina **PPGCC21 – Reconhecimento de
Padrões** (PPGCC-CM, UTFPR Campo Mourão). O projeto modela o **churn voluntário
de diaristas** de uma plataforma de serviços domésticos sob demanda como um
problema de **tempo-até-evento (análise de sobrevivência)**, transferindo para
esse domínio o paradigma de Puram et al. (2026).

Três modelos são comparados sob um protocolo rigoroso: **Cox PH** (linha de
base), **Random Survival Forest (RSF)** e **Gradient Boosting de sobrevivência**.

## Dados: sintético primeiro, real depois

Os dados reais são **proprietários** e não podem ser publicados. Para garantir
a reprodutibilidade pública, o repositório inclui um **gerador de base
sintética** (`src/sintetico.py`) que gera três tabelas ilustrativas do domínio
(`taskers`, `jobs`, `incidents`) e um processo de sobrevivência latente com
propriedades estatísticas plausíveis para o domínio.
A **extração do BigQuery** (queries e integração) fica em um **submódulo privado**
(`private/`), separado justamente por depender do *schema* proprietário e de
credenciais — este repositório público roda ponta a ponta **sem** ele.

Todo o pipeline (construção da cohort → pré-processamento → modelagem →
avaliação) roda **sem alteração** sobre a base sintética ou sobre os dados reais
extraídos do BigQuery: muda apenas a fonte (`--fonte sintetico|bruto`).

## Estrutura

```
config/config.yaml      Sementes e parâmetros (landmark, W, expurgo, busca)
src/
  sintetico.py          Gerador das 3 tabelas sintéticas
  cohort.py             Marcos do ciclo de vida, landmarking, alvo, censura, expurgo
  atributos.py          Engenharia de atributos na janela de landmark
  preprocessamento.py   Pipelines sklearn por modelo (sem vazamento)
  modelos.py            Cox PH, RSF, GBM + espaços de busca
  avaliacao.py          C-index Harrell/Uno, AUC(t), IBS
  estatistica.py        Friedman + Nemenyi + distância crítica
  interpretabilidade.py Permutação, dependência parcial, SHAP
  sensibilidade.py      Varredura de W e landmark
  visualizacao.py       Kaplan-Meier, diferença crítica, importância
experimentos/           Scripts orquestradores 01..06
resultados/             Tabelas (.tex/.csv) e figuras (.pdf) do artigo
private/                Submódulo PRIVADO (extração BigQuery) — opcional, só p/ dados reais
```

## Instalação

Requer Python 3.10–3.12 (o `scikit-survival` ainda não publica wheels para
3.13/3.14). Com [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

ou com pip tradicional:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Execução (ponta a ponta, base sintética)

```bash
python experimentos/01_gerar_sintetico.py        # gera dados/sintetico/
python experimentos/02_construir_cohort.py       # cohort + Kaplan-Meier + descritiva
python experimentos/03_treinar_avaliar.py        # k-fold + métricas + modelos finais
python experimentos/04_comparar_estatistica.py   # Friedman + Nemenyi + diferença crítica
python experimentos/05_interpretar.py            # importância, PDP, SHAP
python experimentos/06_sensibilidade.py          # varredura de W e landmark
```

Acrescente `--rapido` ao experimento 03 para uma validação ágil (menos folds e
iterações). Os artefatos são gravados em `resultados/tabelas/` e
`resultados/figuras/`.

## Execução com os dados reais (BigQuery)

A extração fica no **submódulo privado** `private/` (repositório
`mestrado-churn-de-profissionais-private`). Com acesso ao submódulo e as
credenciais do Google Cloud configuradas (`gcloud auth application-default login`):

```bash
git submodule update --init private                     # monta o submódulo privado
python private/experimentos/00_extrair_bigquery.py      # -> dados/bruto/
python experimentos/02_construir_cohort.py --fonte bruto
python experimentos/03_treinar_avaliar.py
python experimentos/04_comparar_estatistica.py
python experimentos/05_interpretar.py
python experimentos/06_sensibilidade.py --fonte bruto
```

Sem o submódulo (repositório público apenas), use a base sintética (seção acima):
o pipeline roda idêntico, mudando apenas `--fonte`.

## Reprodutibilidade

- **Sementes** fixadas em todas as etapas estocásticas (`config.yaml`, semente única).
- **Configurações** de cada execução registradas em `resultados/execucoes/`.
- **Versões** das dependências fixadas em `requirements.txt`.
- A **janela de observação**, o **corte temporal** e os parâmetros da cohort
  (landmark, W, expurgo) ficam todos em `config.yaml`.

## Mapa: artefatos gerados → seções do artigo

| Artefato (`resultados/`) | Seção / elemento do artigo |
|---|---|
| `tabelas/descritiva_cohort.tex` | Conjunto de Dados (Tabela descritiva) |
| `figuras/kaplan_meier.pdf` | Conjunto de Dados / Resultados (curva de sobrevivência) |
| `tabelas/comparacao_modelos.tex` | Resultados (Tabela comparativa de métricas) |
| `tabelas/hiperparametros.tex` | Materiais e Métodos (espaço de busca / vencedores) |
| `tabelas/teste_friedman.tex`, `tabelas/posthoc_nemenyi.tex` | Resultados (comparação estatística) |
| `figuras/diferenca_critica.pdf` | Resultados (diagrama de diferença crítica) |
| `tabelas/importancia_permutacao.tex`, `figuras/importancia_permutacao.pdf` | Resultados e Discussão (interpretabilidade) |
| `figuras/shap_resumo.pdf`, `tabelas/pdp_*.csv` | Resultados e Discussão (SHAP / dependência parcial) |
| `figuras/km_estratificado.pdf` | Resultados e Discussão (KM estratificado + log-rank: hipótese de desintermediação) |
| `tabelas/sensibilidade.tex` | Resultados e Discussão (robustez a W e landmark) |

## Referência principal

Puram, P., Roy, S., & Gurumurthy, A. (2026). *Factors affecting rider churn in
on-demand food delivery services: insights using a survival analysis and
interpretable machine learning approach.* Journal of the Operational Research
Society. https://doi.org/10.1080/01605682.2026.2656355
