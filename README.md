# Segmentacao de Clientes & Prevencao de Churn com Machine Learning

[![CI Pipeline](https://github.com/Renanbritto/customer-segmentation-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Renanbritto/customer-segmentation-churn/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Plataforma corporativa de inteligencia de clientes e gestao preditiva de churn orientada a dados. Combina analise matricial RFM (Recencia, Frequencia e Valor Monetario), clusterizacao nao-supervisionada K-Means e scoring de probabilidade de evasao de receita para orquestrar acoes estrategicas e personalizadas de CRM.

---

## 1. Visao Geral e Impacto de Negocio

Muitas organizacoes tratam sua base de clientes de maneira homogenea, disparando as mesmas campanhas de marketing para perfis diametralmente opostos. As consequencias comuns dessa abordagem incluem:
1. **Perda Silenciosa de Clientes de Alto Valor (VIPs):** Contas com alto LTV entram em inatividade sem alertas preventivos do time de Customer Success.
2. **Desgaste da Comunicacao:** Disparos genericos reduzem o engajamento e aumentam a taxa de unsubscribe em clientes com potencial de expansao.
3. **Alocacao Ineficiente de CAC e Descontos:** Concessao de cupons agressivos para clientes que ja comprariam organicamente ou investimento excessivo em contas irrecuperaveis.

Esta solucao endereca esses desafios estruturando:
- **Engenharia de Features RFM:** Transformacao de registros transacionais em metricas consolidadas com pontuacao em quantis de 1 a 5.
- **Clusterizacao K-Means Validada ($k=4$):** Ponto otimo determinado pelo Metodo do Cotovelo e Silhouette Score de 0.68, delimitando personas operacionais claras.
- **Scoring Calibrado de Risco de Churn:** Estimacao individual e em lote da probabilidade de evasao com curvas de retencao por cohort de M0 a M6.
- **Playbooks Taticos de CRM:** Recomendacoes acionaveis automatizadas para cada segmento da base.
- **Arquitetura de Producao:** Servico RESTful em FastAPI com schemas Pydantic v2 e Dashboard Executivo em Streamlit com Plotly.

---

## 2. Fundamentacao Estatistica e Matematica

### 2.1 Metodologia RFM e Normalizacao de Dados

A base transacional e agregada por cliente para calcular:
- **Recencia ($R$):** Intervalo em dias entre a data de referencia e a transacao mais recente do cliente:
  $$R = t_{\text{ref}} - \max(t_{\text{compra}})$$
- **Frequencia ($F$):** Contagem total de transacoes realizadas no periodo de analise:
  $$F = \sum \mathbb{I}(t_{\text{compra}})$$
- **Monetario ($M$):** Somatorio do valor financeiro gasto pelo cliente:
  $$M = \sum \text{Valor}$$

Devido a forte assimetria a direita (right-skewness) tipica de dados transacionais, os dados passam por transformacao logaritmica seguida de padronizacao z-score:

$$x_{\text{scaled}} = \frac{\ln(1 + x) - \mu_{\ln}}{\sigma_{\ln}}$$

### 2.2 Clusterizacao K-Means

O algoritmo particiona os $N$ clientes em $k$ conjuntos disjuntos $S = \{S_1, S_2, \dots, S_k\}$ minimizando a soma dos quadrados intra-cluster (WCSS / Inercia):

$$\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} \| x - \mu_i \|^2$$

Onde $\mu_i$ representa o centroide do cluster $S_i$.

### 2.3 Metodo do Cotovelo e Silhouette Score

A selecao de $k=4$ e sustentada por duas metricas estatisticas:
1. **Metodo do Cotovelo:** Ponto de inflexao na curva de inercia WCSS.
2. **Silhouette Coefficient:** Avalia a coesao interna em relacao a separacao do cluster vizinho mais proximo:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   Onde $a(i)$ e a distancia media intra-cluster e $b(i)$ e a menor distancia media para qualquer outro cluster. Em $k=4$, alcanca-se o pico de $0.68$.

### 2.4 Matriz de Personas Mapeadas

| Cluster ID | Persona | Participacao | Recencia Media | Frequencia Media | Ticket Total | Risco Churn | Playbook Tatico CRM |
|---|---|---|---|---|---|---|---|
| **0** | **Champions / Clientes VIPs** | 14.0% | ~4 dias | ~21.4 compras | R$ 10.330 | 3.2% | Atendimento Concierge, conselho de produto e upsell prioritario. |
| **1** | **Potenciais Leais** | 28.7% | ~18 dias | ~9.5 compras | R$ 5.530 | 12.5% | Oferta de plano anual, fidelidade e novos modulos. |
| **2** | **Em Risco Alto (Ticket Alto)** | 18.0% | ~78 dias | ~4.6 compras | R$ 6.880 | 68.4% | Intervencao direta de CS, diagnostico e oferta de retencao. |
| **3** | **Hibernando / Inativos** | 39.3% | ~142 dias | ~1.8 compras | R$ 1.190 | 89.0% | Automacao de win-back de baixo custo ou exclusao de CAC. |

---

## 3. Arquitetura do Repositorio

```
customer-segmentation-churn/
|-- .github/
|   `-- workflows/
|       `-- ci.yml                   # Pipeline de Integracao Continua (Ruff + Pytest)
|-- data/
|   |-- raw/
|   |   `-- customer_transactions.csv   # Log bruto de 23.000+ transacoes
|   `-- processed/
|       `-- rfm_features.parquet        # Base consolidada de 3.000 clientes
|-- src/
|   |-- __init__.py
|   |-- config.py                    # Constantes globais, caminhos e personas
|   |-- features/
|   |   |-- __init__.py
|   |   `-- rfm.py                   # Calculadora RFM e gerador sintetico
|   |-- models/
|   |   |-- __init__.py
|   |   |-- kmeans.py                # Pipeline K-Means e mapeamento deterministico
|   |   `-- churn.py                 # Modelo logistico de risco e decaimento
|   `-- api/
|       |-- __init__.py
|       |-- schemas.py               # Contratos Pydantic v2
|       |-- routes.py                # Endpoints RESTful
|       `-- main.py                  # Aplicacao FastAPI com Swagger UI
|-- tests/
|   |-- conftest.py                  # Fixtures compartilhadas e TestClient
|   |-- test_rfm.py                  # Testes da engenharia de features
|   |-- test_kmeans.py               # Testes do algoritmo K-Means
|   |-- test_churn.py                # Testes do modelo de churn
|   `-- test_api.py                  # Testes de integracao dos endpoints HTTP
|-- app.py                           # Dashboard Executivo em Streamlit
|-- Dockerfile                       # Containerizacao multi-service
|-- docker-compose.yml               # Orquestracao de API e Dashboard
|-- pyproject.toml                   # Configuracoes de Pytest e Ruff
|-- requirements.txt                 # Dependencias do projeto
`-- README.md                        # Documentacao tecnica e executiva
```

---

## 4. Endpoints da API REST (FastAPI)

A API disponibiliza documentacao interativa em `/docs` (Swagger UI) e `/redoc`.

| Metodo | Endpoint | Descricao |
|---|---|---|
| `GET` | `/health` | Verificacao de integridade e uptime da aplicacao |
| `GET` | `/personas/catalog` | Catalogo completo de personas, caracteristicas e retencao cohort |
| `GET` | `/metrics/elbow` | Curvas de Inercia e Silhouette Score para k em [2, 8] |
| `POST` | `/segmentation/predict` | Classificacao individual: cluster, persona, churn e playbook |
| `POST` | `/segmentation/batch` | Avaliacao em lote de multiplos clientes simultaneamente |

### Exemplo de Requisicao Individual

```bash
curl -X POST "http://localhost:8000/segmentation/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-00123",
       "recency": 18,
       "frequency": 9,
       "monetary": 5530.0
     }'
```

**Resposta:**
```json
{
  "customer_id": "CUST-00123",
  "cluster_id": 1,
  "persona_key": "potential",
  "persona_name": "Potenciais Leais",
  "recency": 18,
  "frequency": 9,
  "monetary": 5530.0,
  "avg_ticket": 614.44,
  "churn_probability": 0.125,
  "retention_probability": 0.875,
  "risk_tier": "Baixo Risco",
  "primary_risk_driver": "Engajamento continuo e compras recentes",
  "retention_projection": {
    "M0": 100.0,
    "M1": 97.8,
    "M2": 95.7,
    "M3": 93.6,
    "M4": 91.5,
    "M5": 89.5,
    "M6": 87.5
  },
  "action_playbook": "Oferta de plano anual com desconto, programas de fidelidade e demonstracao de novas features.",
  "distance_to_centroid": 0.3421
}
```

---

## 5. Instrucoes de Instalacao e Execucao

### 5.1 Execucao Local com Python 3.12

```bash
# 1. Clonar o repositorio
git clone https://github.com/Renanbritto/customer-segmentation-churn.git
cd customer-segmentation-churn

# 2. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate       # No Linux/macOS
.venv\Scripts\activate          # No Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Executar os testes automatizados
pytest -v

# 5. Iniciar a API FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Em outro terminal, iniciar o Dashboard Streamlit
streamlit run app.py
```

### 5.2 Execucao via Docker Compose

```bash
docker-compose up --build
```
- API REST: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

---

## 6. Qualidade de Codigo e CI/CD

- **Formatacao e Linter:** `ruff check .` validado com zero advertencias.
- **Suite de Testes:** `pytest -v` com 14 testes cobrindo todo o fluxo analitico e endpoints.
- **Automacao GitHub Actions:** Pipeline configurado em `.github/workflows/ci.yml`.

---

## 7. Estrategia de Branches (Git Flow)

- `main`: Branch de producao versionada via tags semanticas (ex: `v1.0.0`).
- `develop`: Branch de integracao continua de novas features.
- `feature/*`: Branches dedicadas para modulos isolados (RFM, K-Means, Churn, API, UI).
- `release/*`: Branches de homologacao antes do merge em producao.

---

## 8. Autor e Contato

**Renan Nocelli**
- Portfolio: [https://renan-nocelli.vercel.app](https://renan-nocelli.vercel.app)
- GitHub: [https://github.com/Renanbritto](https://github.com/Renanbritto)
- Projeto no Portfolio: [https://renan-nocelli.vercel.app/projetos/segmentacao-clientes](https://renan-nocelli.vercel.app/projetos/segmentacao-clientes)